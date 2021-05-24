""" Harmonic Power Flow using Norton-Raphson

rewriting the harmonic coupled norton equivalent method in a generalized and
modularized way

requires Python 3.5 upwards (uses @ as __matmul__)

fundamental pf largely based on PyPsa implementation (accessed 01.03.2021):
https://github.com/PyPSA/PyPSA/blob/d05b22553403e69e8155fb06cf70618bf9737bf3/pypsa/pf.py#L420

n buses total (i = 1, ..., n)
slack bus is first bus (i = 1)
m-1 linear buses (i = 1, ..., m-1)
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

# start timing
t_start = time.perf_counter()

# global variables TODO: import from config file
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
H_MAX = 5
HARMONICS = [h for h in range(1, H_MAX+1, 2)]
#HARMONICS = [1, 5]
NET_FREQ = 50
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]

# helper definitions
idx = pd.IndexSlice

# pu system
base_current = 1000*BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE


# change complex numbers from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# create infrastructure
def init_lines_from_csv(filename):
    df = pd.read_csv(filename, delimiter=";")
    return df


def init_lines_manually():
    lines = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                                   [2, 2, 3, 0.02, 0.08],
                                   [3, 3, 4, 0.01, 0.02],
                                   [4, 4, 5, 0.01, 0.02],
                                   [5, 5, 6, 0.01, 0.02],
                                   [6, 6, 7, 0.02, 0.08],
                                   [7, 7, 8, 0.01, 0.02],
                                   [8, 8, 9, 0.01, 0.02],
                                   [9, 9, 10, 0.1, 0.02],
                                   [10, 10, 11, 0.02, 0.08],
                                   [11, 11, 12, 0.01, 0.02],
                                   [12, 12, 13, 0.01, 0.02],
                                   [13, 13, 14, 0.1, 0.02],
                                   [14, 14, 15, 0.02, 0.08],
                                   [15, 15, 16, 0.01, 0.02],
                                   [16, 16, 17, 0.01, 0.2],
                                   [17, 17, 18, 0.01, 0.02],
                                   [18, 18, 19, 0.01, 0.2],
                                   [19, 19, 20, 0.01, 0.02],
                                   [20, 20, 1, 0.02, 0.08],
                                   [21, 3, 14, 0.01, 0.02],
                                   [22, 4, 15, 0.01, 0.02],
                                   [23, 1, 11, 0.01, 0.02]]),
                         columns=["ID", "fromID", "toID", "R", "X"])
    return lines


def init_buses_from_csv(filename):
    df = pd.read_csv(filename, delimiter=";")
    return df


def init_buses_manually():
    # df for constant properties of buses
    buses_const = pd.DataFrame(
        np.array([[1, "slack", "generator", 1000, 0.0001],
                 [2, "PQ", "lin_load_1", None, 0],
                 [3, "PQ", "lin_load_2", None, 0],
                 [4, "PQ", "lin_load_3", None, 0],
                 [5, "PQ", "lin_load_4", None, 0],
                 [6, "PQ", "lin_load_5", None, 0],
                 [7, "PQ", "lin_load_6", None, 0],
                 [8, "PQ", "lin_load_7", None, 0],
                 [9, "PQ", "lin_load_8", None, 0],
                 [10, "PQ", "lin_load_9", None, 0],
                 [11, "PQ", "lin_load_10", None, 0],
                 [12, "PQ", "lin_load_11", None, 0],
                 [13, "PQ", "lin_load_12", None, 0],
                 [14, "nonlinear", "smps", None, 0],
                 [15, "nonlinear", "smps", None, 0],
                 [16, "nonlinear", "smps", None, 0],
                 [17, "nonlinear", "smps", None, 0],
                 [18, "nonlinear", "smps", None, 0],
                 [19, "nonlinear", "smps", None, 0],
                 [20, "nonlinear", "smps", None, 0]]),
        columns=["ID", "type", "component", "S", "X_shunt"])

    # # df for real and reactive power of buses
    buses_power = pd.DataFrame(np.zeros((len(buses_const), 2)),
                               columns=["P", "Q"])
    # # insert fundamental powers, part of future import
    buses_power["P"] = [0, 100, 100, 150, 250, 0, 100, 100, 150, 250, 0,
                         100, 100, 150, 250, 0, 100, 100, 150, 250]
    buses_power["Q"] = [0, 100, 100, 100, 100, 0, 100, 100, 100, 100, 0,
                         100, 100, 100, 100, 0, 100, 100, 100, 100]
    # combined df for buses
    buses = pd.concat([buses_const, buses_power], axis=1)
    return buses


def init_network(name, from_csv=True):
    if from_csv:
        buses = init_buses_from_csv(name + "_buses.csv")
        lines = init_lines_from_csv(name + "_lines.csv")
    else:
        buses = init_buses_manually()
        lines = init_lines_manually()
    # find first nonlinear bus
    if len(buses.index[buses["type"] == "nonlinear"]) > 0:
        m = min(buses.index[buses["type"] == "nonlinear"])
    else:
        m = len(buses)
    n = len(buses)
    return buses, lines, m, n


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
    # reactance scales lin. with harmonic no. (Fuchs p.598) (good assumption?)
    for h in harmonics:
        Y = np.zeros([len(buses), len(buses)], dtype=complex)
        # non-diagonal elements
        for ix, line in lines.iterrows():
            Y[int(line.fromID - 1), int(line.toID - 1)] = \
                -1/(line.R + 1j*line.X*h)
            # admittance matrix is assumed to be symmetric
            Y[int(line.toID - 1), int(line.fromID - 1)] = \
                Y[int(line.fromID - 1), int(line.toID - 1)]
        # slack self admittance added as subtransient(?) admittance (p.288/595)
        for n in range(len(buses)):
            if buses["X_shunt"][n] != 0 and h != 1:
                Y[n, n] = -sum(Y[n, :]) + 1/(1j*buses["X_shunt"][n]*h)
            else:
                Y[n, n] = -sum(Y[n, :])
        Y_all.loc[h] = Y
    return Y_all


def init_voltages(buses, harmonics):
    iterables = [harmonics, buses.index.values]
    multi_idx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
    V = pd.DataFrame(np.zeros((len(harmonics) * len(buses), 2)),
                     index=multi_idx, columns=["V_m", "V_a"])
    V.sort_index(inplace=True)
    # set standard initial voltage magnitudes (in p.u.)
    V.loc[1, "V_m"] = 1
    if len(harmonics) > 1:
        V.loc[harmonics[1]:, "V_m"] = 0.1
    return V


def init_fund_state_vec(V):
    # following PyPSA convention instead of Fuchs by not alternating between
    # voltage angle and magnitude
    x = np.append(V.loc[(1, "V_a")][1:], V.loc[(1, "V_m")][1:])
    return x


def fund_mismatch(buses, V, Y1):
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    S = (buses["P"] + 1j*buses["Q"])/BASE_POWER
    mismatch = np.array(V_vec*np.conj(Y1.dot(V_vec)) + S, dtype="c16")
    # again following PyPSA conventions
    f = csr_matrix(np.r_[mismatch.real[1:], mismatch.imag[1:]])
    err = abs(f).max()
    return f, err


def build_jacobian(V, Y1):
    """ fundamental Jacobian containing partial derivatives of S wrt V """
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = diags(Y1 @ V_vec)
    V_diag = diags(V_vec)
    V_diag_norm = diags(V_vec/abs(V_vec))

    dSdt = 1j*V_diag @ (np.conj(I_diag - Y1 @ V_diag))
    dSdV = V_diag_norm @ np.conj(I_diag) + \
        V_diag @ np.conj(Y1 @ V_diag_norm)

    # divide sub-matrices into real and imag part, cut off slack, build J
    dPdt = csr_matrix(dSdt[1:, 1:].real)
    dPdV = csr_matrix(dSdV[1:, 1:].real)
    dQdt = csr_matrix(dSdt[1:, 1:].imag)
    dQdV = csr_matrix(dSdV[1:, 1:].imag)
    J = vstack([hstack([dPdt, dPdV]),
                hstack([dQdt, dQdV])], format="csr")
    return J


def update_fund_state_vec(J, x, f):
    """ perform Newton-Raphson iteration """
    # TODO: try other solvers
    x_new = x - spsolve(J, f.T)
    return x_new


def update_fund_voltages(V, x):
    V.loc[idx[1, 1:], "V_a"] = x[:int(len(x)/2)]
    V.loc[idx[1, 1:], "V_m"] = x[int(len(x)/2):]
    # avoid negative voltage magnitudes (to be able to norm with abs())
    V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change signum
    V["V_a"] = V["V_a"] % (2*np.pi)  # modulo phase wrt 2pi
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
        print("Maximum of " + str(n_iter_f) + " iterations reached.")
    return V, err_t, n_iter_f


def import_Norton_Equivalents(buses, coupled):
    """import Norton Equivalents from files, returns dict of I_N, Y_N pairs"""
    # TODO: import NEs from other sources or input manually
    #  alternative format? (e.g. HDF5 instead of csv)

    NE = {}
    nl_components = buses.component[buses.type == "nonlinear"].unique()
    for device in nl_components:
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
    S = buses.loc[1:(m-1), "P"]/BASE_POWER + \
        1j*buses.loc[1:(m-1), "Q"]/BASE_POWER
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
    f = np.concatenate([f_c.real, f_c.imag])
    err_h = np.linalg.norm(f, np.Inf)
    return f, err_h


def harmonic_state_vector(V):
    """ returns voltages vector, magnitude then phase, without slack at h=1 """
    x = np.append(V.V_m[1:], V.V_a[1:])
    return x


def build_harmonic_jacobian(V, Y, NE, coupled):
    # some arrays to simplify calculation
    V_vec = V.V_m*np.exp(1j*V.V_a)
    V_diag = diags(np.array(V_vec))
    V_norm = V_vec/V.V_m
    V_norm_diag = diags(np.array(V_norm))
    Y_diag = block_diag([np.array(Y.loc[i]) for i in HARMONICS], format="csr")

    # IV and IT, convert to lil_matrix for more efficient element addition
    IV = (Y_diag @ V_norm_diag).tolil()  # diagonal blocks for p = h
    IT = (1j*Y_diag @ V_diag).tolil()

    # iterate through IV and subtract derived current injections
    # number of harmonics (without fundamental)
    K = len(HARMONICS) - 1
    n_blocks = K+1
    # indices of first nonlinear bus at each harmonic
    nl_idx_start = list(range(m, n*(K+1), n))
    # indices of all nonlinear buses
    nl_idx_all = sum([list(range(nl, nl+n-m)) for nl in nl_idx_start], [])
    nl_V = V_vec.iloc[nl_idx_all]
    nl_V_norm = nl_V/A2P(nl_V)[0]
    # Fuchs didn't derive the current injections, so maybe I shouldn't either?
    if coupled:
        for h in range(n_blocks):  # iterating through blocks vertically
            for p in range(n_blocks):   # ... and horizontally
                for i in range(m, n):  # iterating through nonlinear buses
                    # within NE "[1]" points to Y_N
                    Y_N = NE[buses.loc[i].component][1]
                    # subtract derived current injections at respective idx
                    IV[h*n+i, p*n+i] -= Y_N.iloc[h, p] * \
                                            nl_V_norm[(HARMONICS[p], i)]
                    IT[h*n+i, p*n+i] -= 1j*Y_N.iloc[h, p] * \
                                            nl_V[(HARMONICS[p], i)]
    else:
        for h in range(n_blocks):  # iterating through blocks diagonally (p=h)
            for i in range(m, n):  # iterating through nonlinear buses
                # within NE "[1]" points to Y_N
                Y_N = NE[buses.loc[i].component][1]
                # Y_N is one-dimensional for uncoupled case
                IV[h*n+i, h*n+i] -= Y_N.iloc[0, h]*nl_V_norm[(HARMONICS[h], i)]
                IT[h*n+i, h*n+i] -= 1j*Y_N.iloc[0, h]*nl_V[(HARMONICS[h], i)]
    # crop
    IV = IV[m:, 1:]
    IT = IT[m:, 1:]

    # SV and ST (from fundamental)
    # TODO: Harmonize sorting with fundamental
    Y1 = csr_matrix(Y.loc[1])
    V_vec_1 = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = diags(Y1 @ V_vec_1)
    V_diag = diags(V_vec_1)
    V_diag_norm = diags(V_vec_1/V.loc[1, "V_m"])

    S1V1 = V_diag_norm @ np.conj(I_diag) \
        + V_diag @ np.conj(Y1 @ V_diag_norm)
    S1T1 = 1j*V_diag @ (np.conj(I_diag - Y1 @ V_diag))

    SV = hstack([S1V1[1:m, 1:], np.zeros((m-1, n*K))])
    ST = hstack([S1T1[1:m, 1:], np.zeros((m-1, n*K))])

    J = vstack([hstack([SV.real, ST.real]),
                hstack([IV.real, IT.real]),
                hstack([SV.imag, ST.imag]),
                hstack([IV.imag, IT.imag])], format="csr")
    return J


def update_harmonic_state_vec(J, x, f):
    """ perform Newton-Raphson iteration """
    x_new = x - spsolve(J, f)
    return x_new


def update_harmonic_voltages(V, x):
    """update and clean all voltages after iteration"""
    V.iloc[idx[1:], 0] = x[:int(len(x)/2)]
    V.iloc[idx[1:], 1] = x[int(len(x)/2):]

    # avoid negative voltage magnitudes
    # add pi to negative voltage magnitudes
    # V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    # V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change sign
    # -> this doesn't work, why? Instead only performed once in the end
    # -> normalization will give false negative result if doing this

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
    NE = import_Norton_Equivalents(buses, coupled)
    t_end_NE_import = time.perf_counter()
    n_iter_h = 0
    f, err_h = harmonic_mismatch(V, Y, buses, NE)
    x = harmonic_state_vector(V)
    # import all NE of nonlinear devices present in "buses"
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
        print("Maximum of " + str(n_iter_h) + " iterations reached.")
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


buses, lines, m, n = init_network("net2")
V_h, err_h_final, n_iter_h, J = hpf(buses, lines, coupled=True,
                                    plt_convergence=False)
THD_buses = get_THD(V_h)


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

# if __name__ == '__main__':
