""" Harmonic Power Flow using Norton-Raphson

rewriting the harmonic coupled norton equivalent method in a generalized and
modularized way

requires Python 3.5 upwards (uses @ as __matmul__)

fundamental pf largely based on PyPsa implementation (accessed 01.03.2021):
https://github.com/PyPSA/PyPSA/blob/d05b22553403e69e8155fb06cf70618bf9737bf3/pypsa/pf.py#L420

n buses total (i = 1, ..., n)
slack bus is first bus (i = 1)
c PV buses (including slack, i = 1, ..., c)
m-1 linear buses (i = 1, ..., m-1)  -> incorrect, index confusion happened
n-m+1 nonlinear buses (i = m, ..., n)
K harmonics considered (excluding fundamental)
L is last harmonic considered
"""

# TODO: unify writing harmonics as frequency or multiple of fundamental freq.
#  variable naming convention
#  cleaner way of converting between panda and numpy objects


import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
#from scipy.linalg import solve, block_diag as block_diag_dense
from scipy.sparse import diags, csr_matrix, hstack, vstack, block_diag
import matplotlib.pyplot as plt
from sys import getsizeof
import time


# change complex numbers from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# create infrastructure
def init_lines_from_csv(filename):
    """ Import line data from .csv file.
    
    DataFrame with columns "ID", "fromID", "toID", "R", "X", "G", "B"

    - IDs start from 1
    - R + jX is the lines serial impedance [Ohm]
    - G + jX is the lines shunt admittance [S] = [1/Ohm]
    """
    df = pd.read_csv(filename, delimiter=";")

    # convert to pu system
    df.loc[:, "R"] = df.R/base_impedance
    df.loc[:, "X"] = df.X/base_impedance
    df.loc[:, "G"] = df.G/base_admittance
    df.loc[:, "B"] = df.B/base_admittance
    return df


def init_lines_manually():
    lines = pd.DataFrame(np.array([[1, 1, 2, 0.5, 0.5, 0, 0.05],
                                   [2, 2, 3,   1,   4, 0, 0.1],
                                   [3, 3, 4, 0.5,   1, 0, 0.05],
                                   [4, 4, 1, 0.5,   1, 0, 0.05]]),
                         columns=["ID", "fromID", "toID", "R", "X", "G", "B"])
    lines.loc[:, "R"] = lines.R/base_impedance
    lines.loc[:, "X"] = lines.X/base_impedance
    lines.loc[:, "G"] = lines.G/base_admittance
    lines.loc[:, "B"] = lines.B/base_admittance
    return lines


def init_buses_from_csv(filename):
    """ Import bus data from .csv file.
    
    DataFrame with columns:
    ["ID", "type", "component", "S", "P", "Q", "X_sh"]

    - IDs start from 1 (slack), then list PV, then PQ buses, then nonlinear
    - S = P + jQ is active and reactive Power [W]
      it is negative for power generating devices and positive for loads
    - X_sh is shunt impedance, set to 0 if none present
    """
    df = pd.read_csv(filename, delimiter=";")
    df.loc[:, "S"] = df.S/BASE_POWER
    df.loc[:, "P"] = df.P/BASE_POWER
    df.loc[:, "Q"] = df.Q/BASE_POWER
    df.loc[:, "X_sh"] = df.X_sh/base_impedance
    #df.loc[:, "V_nom"] = df.V_nom/BASE_VOLTAGE
    return df


def init_buses_manually():
    # df for constant properties of buses
    buses = pd.DataFrame(
        np.array([[1, "slack", "generator", 0, 0, 0, 0.005],
                  [2, "PQ", "lin_load_1", 0, 100, 100, 0],
                  [3, "PQ", "lin_load_2", 0, 100, 100, 0],
                  [4, "nonlinear", "smps", 0, 150, 100, 0]]),
        columns=["ID", "type", "component", "S", "P", "Q", "X_sh"])
    buses.loc[:, "S"] = buses.S/BASE_POWER
    buses.loc[:, "P"] = buses.P/BASE_POWER
    buses.loc[:, "Q"] = buses.Q/BASE_POWER
    buses.loc[:, "X_sh"] = buses.X_sh/base_impedance
    buses.loc[:, "V_nom"] = buses.V_nom/BASE_VOLTAGE
    return buses


def init_network(filename_buses, filename_lines, from_csv=True):
    # TODO: sort buses here, so that users don't have to follow sorting rules
    if from_csv:
        buses = init_buses_from_csv(filename_buses)
        lines = init_lines_from_csv(filename_lines)
    else:
        buses = init_buses_manually()
        lines = init_lines_manually()
    # find first nonlinear bus
    if len(buses.index[buses["type"] == "nonlinear"]) > 0:
        m = min(buses.index[buses["type"] == "nonlinear"])
    else:
        m = len(buses)
    n = len(buses)
    c = len(buses[buses.type == "PV"]) + 1
    return buses, lines, m, n, c


# Functions for Fundamental Power Flow
def build_admittance_matrices(buses, lines, harmonics):
    """ Create admittance matrices for all harmonics

    based on infrastructure, that is lines and buses (nodes)
    :returns: multi-index DataFrame, dim = complex(n_buses, n_buses*(K+1))
    """
    # initialize empty harmonic admittance matrices
    iterables = [harmonics, buses.index.values]
    multi_idx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
    Y_all = pd.DataFrame(
        np.zeros((len(harmonics) * len(buses), len(buses))),
        index=multi_idx, columns=[buses.index.values], dtype="c16")

    # Harmonic admittance matrices
    # reactance scales lin. with harmonic no. (Fuchs p.598)
    for h in harmonics:
        Y = np.zeros([len(buses), len(buses)], dtype=complex)
        # non-diagonal elements
        for ix, line in lines.iterrows():
            Y[int(line.fromID - 1), int(line.toID - 1)] = \
                -1/(line.R + 1j*line.X*h)
            # admittance matrix is assumed to be symmetric
            Y[int(line.toID - 1), int(line.fromID - 1)] = \
                Y[int(line.fromID - 1), int(line.toID - 1)]
        # self admittances added as subtransient(?) admittance (p.288/595)
        for n in range(len(buses)):
            if buses["X_sh"][n] != 0 and h != 1:
                Y[n, n] = -sum(Y[n, :]) + 1/(1j*buses["X_sh"][n]*h)
            else:
                Y[n, n] = -sum(Y[n, :])
            # Adding shunt admittances for each pi-model line connected to bus n
            for m in range(len(lines)):
                # go through all lines, see if they are connected to bus n,
                # if yes add shunt admittance of pi line to bus admittance
                if lines.loc[m, "fromID"] == n or lines.loc[m, "toID"] == n:
                    Y[n, n] = Y[n, n] + \
                              (lines.loc[m, "G"]+1j*h*lines.loc[m, "B"])/2

        Y_all.loc[h] = Y
    return Y_all


def init_voltages(buses, harmonics):
    iterables = [harmonics, buses.index.values]
    multi_idx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
    V = pd.DataFrame(np.zeros((len(harmonics) * len(buses), 2)),
                     index=multi_idx, columns=["V_m", "V_a"])
    V.sort_index(inplace=True)
    # set standard initial voltage magnitudes (in p.u.)
    V.loc[1, "V_m"] = 1  # TODO: use V_nom?
    if len(harmonics) > 1:
        V.loc[harmonics[1]:, "V_m"] = 0.1
    return V


def init_fund_state_vec(V):
    # following PyPSA convention instead of Fuchs by not alternating between
    # voltage angle and magnitude
    # V_a for PQ and PV buses, V_m only for PQ buses
    x = np.append(V.loc[(1, "V_a")][1:], V.loc[(1, "V_m")][c:])
    return x


def fund_mismatch(buses, V, Y1):
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    S = (buses["P"] + 1j*buses["Q"])
    mismatch = np.array(V_vec*np.conj(Y1.dot(V_vec)) + S, dtype="c16")
    # again following PyPSA conventions
    f = csr_matrix(np.r_[mismatch.real[1:], mismatch.imag[c:]])
    err = abs(f).max()
    return f, err


def build_jacobian(V, Y1):
    """ fundamental Jacobian containing partial derivatives of S wrt V """
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = diags(Y1 @ V_vec)
    V_diag = diags(V_vec)
    V_diag_norm = diags(V_vec/abs(V_vec))

    dSdA = 1j*V_diag @ (np.conj(I_diag - Y1 @ V_diag))
    dSdV = V_diag_norm @ np.conj(I_diag) + \
        V_diag @ np.conj(Y1 @ V_diag_norm)

    # divide sub-matrices into real and imag part, cut off slack, build J
    dPdA = csr_matrix(dSdA[1:, 1:].real)
    dPdV = csr_matrix(dSdV[1:, c:].real)
    dQdA = csr_matrix(dSdA[c:, 1:].imag)
    dQdV = csr_matrix(dSdV[c:, c:].imag)
    J = vstack([hstack([dPdA, dPdV]),
                hstack([dQdA, dQdV])], format="csr")
    return J


def update_fund_state_vec(J, x, f):
    """ perform Newton-Raphson iteration """
    # TODO: find and try other solvers
    x_new = x - spsolve(J, f.T)
    return x_new


def update_fund_voltages(V, x):
    V.loc[idx[1, 1:], "V_a"] = x[0:(n-1)]
    V.loc[idx[1, c:], "V_m"] = x[(n-1):]
    # avoid negative voltage magnitudes (to be able to norm with abs())
    # --> any scenario where I actually need this?
    # V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    # V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change signum
    # V["V_a"] = V["V_a"] % (2*np.pi)  # modulo phase wrt 2pi
    return V


def pf(Y, buses, thresh_f = 1e-6, max_iter_f = 30, plt_convergence=False):
    """ execute fundamental power flow

    :param plt_convergence(default=False), shows convergence behaviour by
           plotting err_t
    :return: V: final voltages
             err_t: error over time
             n_iter_f: number of iterations performed
    """
    V = init_voltages(buses, HARMONICS)
    n_iter_f = 0
    Y1 = np.array(Y.loc[1])
    x = init_fund_state_vec(V)
    f, err = fund_mismatch(buses, V, Y1)
    err_t = {}
    while err > thresh_f and n_iter_f < max_iter_f:
        J = build_jacobian(V, Y1)
        x = update_fund_state_vec(J, x, f)
        V = update_fund_voltages(V, x)
        f, err = fund_mismatch(buses, V, Y1)
        err_t[n_iter_f] = err
        n_iter_f += 1
    # plot convergence behaviour
    if plt_convergence:
        plt.plot(list(err_t.keys()), list(err_t.values()))
    print(V.loc[1])
    if n_iter_f < max_iter_f:
        print("Fundamental power flow converged after " + str(n_iter_f) +
              " iterations.")
    elif n_iter_f == max_iter_f:
        print("Warning! Maximum of " + str(n_iter_f) + " iterations reached.")
    return V, err_t, n_iter_f


def import_Norton_Equivalents(buses, coupled):
    """import Norton Equivalents from files, returns dict of I_N, Y_N pairs"""
    # TODO: import NEs from other sources or input manually
    #  find a better way how to choose considered harmonics and correct file
    #  alternative format? (e.g. HDF5 instead of csv)

    NE = {}
    nl_components = buses.component[buses.type == "nonlinear"].unique()
    for device in nl_components:
        #file_path = str("~/Git/harmonic-power-flow/Circuit Simulation/"
        #                + device + "_NE.csv")
        file_path = str("~/Git/harmonic-power-flow/Circuit Simulation/"
                + device + "_NE.csv")
        NE_device = pd.read_csv(file_path, index_col=["Parameter", "Frequency"])
        # change column type from str to int
        NE_device.columns = NE_device.columns.astype(int)
        # filter all columns for harmonics considered
        # TODO: check if all NE for all considered harmonics are available
        NE_device = NE_device[HARMONICS_FREQ]
        # values are imported as strings, transform to complex
        NE_device = NE_device.apply(lambda col: col.apply(
            lambda val: complex(val.strip('()'))))
        # change to pu system
        if coupled:
            I_N = NE_device.loc["I_N_c"]/base_current
            # also filter Y_N_c rows by harmonics considered
            Y_N = NE_device.loc[("Y_N_c", HARMONICS_FREQ), HARMONICS_FREQ] / \
                base_admittance
        else:
            I_N = NE_device.loc["I_N_uc"]/base_current
            Y_N = NE_device.loc["Y_N_uc"]/base_admittance
        NE[device] = [I_N, Y_N]
    return NE


def current_injections(busID, V, NE):
    """calculates the harmonic current injections at one bus"""
    device = buses.loc[busID-1, "component"]
    (I_N, Y_N) = NE[device]
    V_h = V.loc[idx[:, busID-1], "V_m"]*np.exp(1j*V.loc[idx[:, busID-1], "V_a"])
    # coupled: Y_N is a matrix, uncoupled: vector
    if Y_N.shape[0] > 1:
        I_inj = np.squeeze(I_N) - Y_N.dot(V_h.to_numpy()).droplevel(0)
    else:
        I_inj = np.squeeze(I_N) - np.diag(np.squeeze(Y_N)).dot(V_h.to_numpy())
    return I_inj


def current_balance(V, Y, buses, NE):
    """ evaluate current balance at all buses

    Fundamental current balance only for nonlinear buses (n-m+1)
    Harmonic current balance for all buses and all harmonics (n*K)
    :return: vector of n-m+1 + nK complex current balances (as np.array)
    """

    # fundamental admittance for nonlinear buses
    Y_f = csr_matrix(Y.loc[1, m:, :])
    # fundamental voltage for all buses
    V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    # fundamental line currents at nonlinear buses
    dI_f = Y_f @ V_f

    # construct V and Y from list of sub-arrays except fund
    Y_h = block_diag(([np.array(Y.loc[i]) for i in HARMONICS[1:]]), format="csr")
    V_h = V.loc[HARMONICS[1:], "V_m"]*np.exp(1j*V.loc[HARMONICS[1:], "V_a"])
    # harmonic line currents at all buses
    dI_h = Y_h @ V_h

    for i in range(m, n):
        # current injections of all harmonics at bus i
        I_inj = current_injections(buses.ID[i], V, NE)
        # fundamental current balance
        dI_f[i-m] += I_inj[HARMONICS_FREQ[0]]
        # harmonic current balance, add injection at respective index
        for p in range(len(HARMONICS[1:])):
            dI_h[p*n + i] += I_inj[HARMONICS_FREQ[p+1]]
    # final current balance vector
    dI = np.concatenate([dI_f, dI_h])
    return dI


def harmonic_mismatch(V, Y, buses, NE):
    """ power and current mismatches for harmonic power flow

    also referred to as harmonic mismatch vector f_h, that needs to be minimized
    during NR algorithm
    :return: numpy vector of powers (m-2) and currents (n-m+1 + nK),
             first real part, then imaginary part
             total length: 2(m-2 + n-m+1+nK) = 2(n(K+1)-1)
    """

    # fundamental power mismatch
    # add all linear buses to S except slack (# = m-2)
    S = buses.loc[1:(m-1), "P"] + 1j*buses.loc[1:(m-1), "Q"]
    # prepare V and Y as needed
    V_i = V.loc[idx[1, 1:(m-1)], "V_m"] * \
        np.exp(1j*V.loc[idx[1, 1:(m-1)], "V_a"])
    V_j = V.loc[1, "V_m"] * np.exp(1j*V.loc[1, "V_a"])
    Y_ij = csr_matrix(Y.loc[idx[1, 1:(m-1), :]])
    # get rid of indices for calculation
    Sl = (V_i*np.conjugate(Y_ij @ V_j)).to_numpy()
    dS = S.to_numpy() + Sl

    # current mismatch
    dI = current_balance(V, Y, buses, NE)
    # combine both
    f_c = np.concatenate([dS, dI])

    # Convergence: err_h < THRESH_H
    f = np.concatenate([f_c.real, f_c[c-1:].imag])
    err_h = np.linalg.norm(f, np.Inf)
    return f, err_h


def harmonic_state_vector(V):
    """ returns voltages vector, phase then magnitude

    at h=1: without slack and without magnitude for PV buses"""
    x = np.append(V.V_a[1:], V.V_m[c:])
    return x


def build_harmonic_jacobian(V, Y, NE, coupled):
    # some arrays to simplify calculation
    V_vec = V.V_m*np.exp(1j*V.V_a)
    V_diag = diags(np.array(V_vec))
    V_norm = V_vec/V.V_m
    V_norm_diag = diags(np.array(V_norm))
    Y_diag = block_diag([np.array(Y.loc[i]) for i in HARMONICS], format="csr")

    # dIdV and dIdA, convert to lil_matrix for more efficient element addition
    dIdV = (Y_diag @ V_norm_diag).tolil()  # diagonal blocks for p = h
    dIdA = (1j*Y_diag @ V_diag).tolil()

    # iterate through IV and subtract derived current injections
    # number of harmonics (without fundamental)
    K = len(HARMONICS) - 1
    n_blocks = K+1
    # indices of first nonlinear bus at each harmonic
    nl_idx_start = list(range(m, n*(K+1), n))
    # indices of all nonlinear buses
    nl_idx_all = sum([list(range(nl, nl+n-m)) for nl in nl_idx_start], [])
    nl_V = V_vec.iloc[nl_idx_all]
    nl_V_norm = nl_V/V.iloc[nl_idx_all, 0]
    # Fuchs didn't derive the current injections, so maybe I shouldn't either?
    # --> Algorithm is diverging without them.
    if coupled:
        for h in range(n_blocks):  # iterating through blocks vertically
            for p in range(n_blocks):   # ... and horizontally
                for i in range(m, n):  # iterating through nonlinear buses
                    # within NE "[1]" points to Y_N
                    Y_N = NE[buses.loc[i].component][1]
                    # subtract derived current injections at respective idx
                    dIdV[h*n+i, p*n+i] -= Y_N.iloc[h, p] * \
                                            nl_V_norm[(HARMONICS[p], i)]
                    dIdA[h*n+i, p*n+i] -= 1j*Y_N.iloc[h, p] * \
                                            nl_V[(HARMONICS[p], i)]
    else:
        for h in range(n_blocks):  # iterating through blocks diagonally (p=h)
            for i in range(m, n):  # iterating through nonlinear buses
                # within NE "[1]" points to Y_N
                Y_N = NE[buses.loc[i].component][1]
                # Y_N is one-dimensional for uncoupled case
                dIdV[h*n+i, h*n+i] -= Y_N.iloc[0, h]*nl_V_norm[(HARMONICS[h], i)]
                dIdA[h*n+i, h*n+i] -= 1j*Y_N.iloc[0, h]*nl_V[(HARMONICS[h], i)]
    # crop
    dIdA = dIdA[m:, 1:]
    dIdV = dIdV[m:, c:]


    # SV and SA (from fundamental)
    # TODO: Harmonize sorting with fundamental
    Y1 = csr_matrix(Y.loc[1])
    V_vec_1 = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = diags(Y1 @ V_vec_1)
    V_diag = diags(V_vec_1)
    V_diag_norm = diags(V_vec_1/V.loc[1, "V_m"])

    dS1dA1 = 1j*V_diag @ (np.conj(I_diag - Y1 @ V_diag))
    dS1dV1 = V_diag_norm @ np.conj(I_diag) \
        + V_diag @ np.conj(Y1 @ V_diag_norm)
    
    dSdA = csr_matrix(hstack([dS1dA1, np.zeros((n, n*K))]))
    dSdV = csr_matrix(hstack([dS1dV1, np.zeros((n, n*K))]))

    dPdA = dSdA[1:m, 1:].real
    dPdV = dSdV[1:m, c:].real
    dQdA = dSdA[c:m, 1:].imag
    dQdV = dSdV[c:m, c:].imag
    
    J = vstack([hstack([dPdA, dPdV]),
                hstack([dIdA.real, dIdV.real]),
                hstack([dQdA, dQdV]),
                hstack([dIdA.imag, dIdV.imag])], format="csr")
    return J


def update_harmonic_state_vec(J, x, f):
    """ perform Newton-Raphson iteration """
    x_new = x - spsolve(J, f)
    return x_new


def update_harmonic_voltages(V, x):
    """update and clean all voltages after iteration"""
    V.iloc[idx[1:], 1] = x[:(n*len(HARMONICS)-1)]
    V.iloc[idx[c:], 0] = x[(n*len(HARMONICS)-1):]


    # modulo phase wrt 2pi (doesn't work)
    #V["V_a"] = V["V_a"] % (2*np.pi)

    # simulating julia behaviour (also doesn't seam to help)
    # whenever phase is bigger than 2pi, normal modulo
    #V.loc[V.V_a > 2*np.pi, "V_a"] = \
    #    V.loc[V.V_a > 2*np.pi, "V_a"] % 2*np.pi
    # whenever phase is smaller than -2pi, not normal modulo
    #V.loc[V.V_a < -2*np.pi, "V_a"] = \
    #    -1*(abs(V.loc[V.V_a < -2*np.pi, "V_a"]) % 2*np.pi)

    # avoid negative voltage magnitudes
    # add pi to negative voltage magnitudes
    # V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    # V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change sign
    # -> this doesn't work, why? Instead only performed once in the end
    # --> normalization will give false negative result if doing this

    # TODO: calculations (probably normalization) independent of phase modulo

    return V


def hpf(buses, lines, coupled, thresh_h=1e-4, max_iter_h=50,
        plt_convergence=False):
    """ execute fundamental power flow

    :param plt_convergence(default=False), shows convergence behaviour by
           plotting err_t
    :return: V: final voltages
             err_t: error over time
             n_iter_f: number of iterations performed
    """
    global t_end_init, t_end_pf, t_end_NE_import

    Y = build_admittance_matrices(buses, lines, HARMONICS)
    t_end_init = time.perf_counter()
    V, err1_t, n_converged = pf(Y, buses)
    t_end_pf = time.perf_counter()
    # import all NE of nonlinear devices present in "buses"
    NE = import_Norton_Equivalents(buses, coupled)
    t_end_NE_import = time.perf_counter()
    n_iter_h = 0
    f, err_h = harmonic_mismatch(V, Y, buses, NE)
    x = harmonic_state_vector(V)
    err_h_t = {}
    global t_start_hpf_solve, t_end_hpf_solve
    t_start_hpf_solve = time.perf_counter()
    while err_h > thresh_h and n_iter_h < max_iter_h:
        J = build_harmonic_jacobian(V, Y, NE, coupled)
        x = update_harmonic_state_vec(J, x, f)
        V = update_harmonic_voltages(V, x)
        (f, err_h) = harmonic_mismatch(V, Y, buses, NE)
        err_h_t[n_iter_h] = err_h
        n_iter_h += 1
    t_end_hpf_solve = time.perf_counter()

    # getting rid of negative voltage magnitudes:
    # add pi to negative voltage magnitudes
    V.loc[V["V_m"] < 0, "V_a"] += np.pi
    V["V_a"] = V["V_a"] % (2*np.pi)  # modulo phase wrt 2pi
    V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change signum

    # plot convergence behaviour
    if plt_convergence:
        plt.plot(list(err_h_t.keys()), list(err_h_t.values()))
    print(V)
    if n_iter_h < max_iter_h:
        print("Harmonic power flow converged after " + str(n_iter_h) +
              " iterations.")
    elif n_iter_h == max_iter_h:
        print("Warning! Maximum of " + str(n_iter_h) + " iterations reached.")
    return V, err_h, n_iter_h, J


def get_THD(V):
    THD = pd.DataFrame(np.zeros((len(buses), 2)), columns=["THD_F", "THD_R"])
    for bus in buses.ID:
        THD.loc[bus-1, "THD_F"] = \
            np.sqrt(sum((V.loc[idx[3:, bus-1], "V_m"])**2)) /\
            (V.loc[idx[1, bus-1], "V_m"])
        THD.loc[bus-1, "THD_R"] = \
            np.sqrt(sum((V.loc[idx[3:, bus-1], "V_m"])**2)) /\
            np.sqrt(sum((V.loc[idx[:, bus-1], "V_m"])**2))
    return THD

# start timing
t_start = time.perf_counter()

# global variables
BASE_POWER = 1000  # in W
BASE_VOLTAGE = 230  # in V

H_MAX = 5  # has to be either 5, 21, 51 or 101

NET_FREQ = 50
HARMONICS = [h for h in range(1, H_MAX+1, 2)]
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]

# helper definitions
idx = pd.IndexSlice

# pu system
base_current = BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE
base_impedance = 1/base_admittance


buses, lines, m, n, c = init_network("berlin_suburban_buses_217.csv",
                                     "berlin_suburban_lines.csv")
#Y = build_admittance_matrices(buses, lines, HARMONICS)
#V_f, err_f, n_converged_f = pf(Y, buses)


V_h, err_h_final, n_iter_h, J = hpf(buses, lines, coupled=False,
                                    plt_convergence=False)
THD_buses = get_THD(V_h)

#V_m_bus4 = V_h.loc[idx[:, 3], "V_m"]
#plt.bar(HARMONICS, V_m_bus4)

t_end = time.perf_counter()
print("Init execution time: " + str(t_end_init - t_start) + " s")
print("Fundamental Power Flow execution time: " +
      str(t_end_pf - t_end_init) + " s")
print("Norton Parameter import execution time: " +
      str(t_end_NE_import - t_end_pf) + " s")
print("Harmonic Power Flow execution time: " +
      str(t_end - t_end_NE_import) + " s")
print("- Only HPF solve execution time: " +
      str(t_end_hpf_solve - t_start_hpf_solve) + " s")
print("Total execution time: " +
      str(t_end - t_start) + " s")
print("THD [%]: ")

print("THD an Bus 4: " + str(THD_buses.THD_F[3]*100))
# if __name__ == '__main__':
