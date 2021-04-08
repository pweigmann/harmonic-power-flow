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
#  test for multiple nonlinear buses as well as only one nonlinear bus
#  variable naming convention

import numpy as np
import pandas as pd
from scipy.sparse.linalg import *
from scipy.linalg import block_diag as block_diag_dense
from scipy.sparse import diags, csr_matrix, hstack, vstack, block_diag
import matplotlib.pyplot as plt
from sys import getsizeof

# global variables
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
HARMONICS = [1, 5, 7]
NET_FREQ = 50
HARMONICS_FREQ = [NET_FREQ * i for i in HARMONICS]
MAX_ITER_F = 30  # maybe better as argument of pf function
MAX_ITER_H = 30
THRESH_F = 1e-6  # error threshold of fundamental mismatch function
THRESH_H = 1e-4
COUPLED_NE = True  # use Norton parameters of coupled vs. uncoupled model
SPARSE = True

# helper definitions
idx = pd.IndexSlice

# pu system
base_current = 1000*BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE

# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# infrastructure (TODO: import infrastructure from file)
# df for constant properties of buses
buses_const = pd.DataFrame(np.array([[1, "slack", "generator", 1000, 0.0001],
                                  [2, "PQ", "lin_load_1", None, 0],
                                  [3, "PQ", "lin_load_2", None, 0],
                                  [4, "PQ", None, None, 0],
                                  [5, "nonlinear", "smps", None, 0]]),
                           columns=["ID", "type", "component", "S", "X_shunt"])
# generate columns for all frequencies (probably not needed without Fuchs)
# columns = []
# for h in HARMONICS:
#     columns.append("P" + str(h))
#     columns.append("Q" + str(h))
# # df for real and reactive power of buses
buses_power = pd.DataFrame(np.zeros((len(buses_const), 2)),
                           columns=["P1", "Q1"])
# # insert fundamental powers, part of future import
buses_power["P1"] = [0, 100, 100, 0, 250]
buses_power["Q1"] = [0, 100, 100, 0, 100]
# combined df for buses
buses = pd.concat([buses_const, buses_power], axis=1)

# find first nonlinear bus FIXME: start counting from 0 or 1? atm mixed, drop ID
m = min(buses.index[buses["type"] == "nonlinear"])
n = len(buses)

lines = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                                  [2, 2, 3, 0.02, 0.08],
                                  [3, 3, 4, 0.01, 0.02],
                                  [4, 4, 5, 0.01, 0.02],
                                  [5, 5, 1, 0.01, 0.02]]),
                        columns=["ID", "fromID", "toID", "R", "X"])


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
        for idx, line in lines.iterrows():
            Y[int(line.fromID - 1), int(line.toID - 1)] = \
                -1/(line.R + 1j*line.X*h)
            # admittance matrix is assumed to be symmetric
            Y[int(line.toID - 1), int(line.fromID - 1)] = \
                Y[int(line.fromID - 1), int(line.toID - 1)]
        # slack self admittance added as subtransient(?) admittance (p.288/595)
        # TODO: find out if or why self admittance is not applied at fund freq
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


def fund_mismatch(buses, V, Y1, sparse=SPARSE):
    if sparse:
        V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
        S = (buses["P1"] + 1j*buses["Q1"])/BASE_POWER
        mismatch = np.array(V_vec*np.conj(Y1.dot(V_vec)) + S, dtype="c16")
        # again following PyPSA conventions
        f = csr_matrix(np.r_[mismatch.real[1:], mismatch.imag[1:]])
        err = np.linalg.norm(f.toarray(), np.Inf)
    else:
        V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
        S = (buses["P1"] + 1j*buses["Q1"])/BASE_POWER
        mismatch = np.array(V_vec*np.conj(Y1.dot(V_vec)) + S, dtype="c16")
        # again following PyPSA conventions
        f = np.r_[mismatch.real[1:], mismatch.imag[1:]]
        err = np.linalg.norm(f, np.Inf)
    return f, err


def build_jacobian(V, Y1, sparse=SPARSE):
    """ fundamental Jacobian containing partial derivatives of S wrt V """
    if sparse:
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
        J = vstack([hstack([dPdt, dPdV]), hstack([dQdt, dQdV])], format="csr")
    else:
        V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
        I_diag = np.diag(Y1.dot(V_vec))
        V_diag = np.diag(V_vec)
        V_diag_norm = np.diag(V_vec/abs(V_vec))

        dSdt = 1j*V_diag.dot(np.conj(I_diag - Y1.dot(V_diag)))

        dSdV = V_diag_norm.dot(np.conj(I_diag)) \
            + V_diag.dot(np.conj(Y1.dot(V_diag_norm)))

        # divide sub-matrices into real and imag part, cut off slack, build J
        dPdt = dSdt[1:, 1:].real
        dPdV = dSdV[1:, 1:].real
        dQdt = dSdt[1:, 1:].imag
        dQdV = dSdV[1:, 1:].imag
        J = np.vstack([np.hstack([dPdt, dPdV]), np.hstack([dQdt, dQdV])])
    return J


def update_fund_state_vec(J, x, f, sparse=SPARSE):
    """ perform Newton-Raphson iteration """
    # TODO: try other solvers
    if sparse:
        x_new = x - spsolve(J, f.T)
    else:
        x_new = x - spsolve(J, f)
    return x_new


def update_fund_voltages(V, x):
    V.loc[idx[1, 1:], "V_a"] = x[:int(len(x)/2)]
    V.loc[idx[1, 1:], "V_m"] = x[int(len(x)/2):]
    return V


def pf(Y, buses, plt_convergence=False):
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
    while err > THRESH_F and n_iter_f < MAX_ITER_F:
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
    if n_iter_f < MAX_ITER_F:
        print("Fundamental power flow converged after " + str(n_iter_f) +
              " iterations.")
    elif n_iter_f == MAX_ITER_F:
        print("Maximum of " + str(n_iter_f) + " iterations reached.")
    return V, err_t, n_iter_f


# fundamental power flow execution
# Y_h = build_admittance_matrices(buses, lines, HARMONICS)
# V_h, err1_t, n_converged = pf(Y_h, buses)

if HARMONICS == [1]:
    pass
    # exit()


def import_Norton_Equivalents(buses, coupled):
    """import Norton Equivalents from files, returns dict of I_N, Y_N pairs"""
    # TODO: import NEs from other sources or input manually
    #  alternative format? (e.g. HDF5 instead of csv)

    NE = {}
    nl_components = buses.component[buses.type == "nonlinear"]
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
    # FIXME: Bug that results in immediate convergence in uncoupled case
    if COUPLED_NE:
        I_inj = np.squeeze(I_N) - Y_N.dot(V_h.to_numpy()).droplevel(0)  # not nice
    else:
        I_inj = np.squeeze(I_N) - Y_N.dot(V_h.to_numpy())
    return I_inj


def current_balance(V, Y, buses, NE, sparse=SPARSE):
    """ evaluate current balance

    Fundamental current balance only for nonlinear buses (n-m+1)
    Harmonic current balance for all buses and all harmonics (n*K)
    :return: vector of n-m+1 + nK complex current balances (as np.array)
    """
    if sparse:
        # fundamental admittance for nonlinear buses
        Y_f = csr_matrix(np.squeeze(Y.loc[1, m:]))
        # fundamental voltage for all buses
        V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
        # fundamental line currents at nonlinear buses
        dI_f = Y_f @ V_f

        # construct V and Y from list of sub-arrays except fund
        # TODO: test for multiple nl buses
        Y_h = block_diag(([Y.loc[i] for i in HARMONICS[1:]]), format="csr")
        V_h = V.loc[HARMONICS[1:], "V_m"]*np.exp(1j*V.loc[HARMONICS[1:], "V_a"])
        # harmonic line currents at all buses
        dI_h = Y_h @ V_h

        for i in range(m, n):
            # current injections of all harmonics at bus i
            I_inj = current_injections(buses.ID[i], V, NE)
            # fundamental current balance
            dI_f -= I_inj[HARMONICS_FREQ[0]]
            # harmonic current balance, subtract injection at appropriate index
            for p in range(len(HARMONICS[1:])):
                dI_h[p*n + m] -= I_inj[HARMONICS_FREQ[p+1]]

        # final current balance vector
        dI = np.array([dI_f, *dI_h])
    else:
        # fundamental admittance for nonlinear buses
        Y_f = Y.loc[1, m:]
        # fundamental voltage for all buses
        V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
        # fundamental line currents at nonlinear buses
        dI_f = np.squeeze(Y_f).dot(V_f)

        # construct V and Y from list of sub-arrays except fund
        #  TODO: test for multiple nl buses
        Y_h = block_diag_dense(*[Y.loc[i] for i in HARMONICS[1:]])
        V_h = V.loc[HARMONICS[1:], "V_m"]*np.exp(1j*V.loc[HARMONICS[1:], "V_a"])
        # harmonic line currents at all buses
        dI_h = Y_h.dot(V_h)

        for i in range(m, n):
            # current injections of all harmonics at bus i
            I_inj = current_injections(buses.ID[i], V, NE)
            # fundamental current balance
            dI_f -= I_inj[HARMONICS_FREQ[0]]
            # harmonic current balance, subtract injection at appropriate index
            for p in range(len(HARMONICS[1:])):
                dI_h[p*n + m] -= I_inj[HARMONICS_FREQ[p+1]]

        # final current balance vector
        dI = np.array([dI_f, *dI_h])
    return dI


def harmonic_mismatch(V, Y, buses, NE):
    """ power and current mismatches for harmonic power flow

    also referred to as harmonic mismatch vector f_h, that needs to be minimized
    during NR algorithm
    :return: numpy vector of powers (m-2) and currents (n-m+1 + nK),
             first real part, then imaginary part
             total length: 2(m-2 + n-m+1+nK) = 2(n(K+1)-1)
    """

    # fundamental power mismatch, first iteration same as in fundamental pf: f
    # add all linear buses to S except slack (# = m-2)
    S = buses.loc[1:(m-1), "P1"]/BASE_POWER + \
        1j*buses.loc[1:(m-1), "Q1"]/BASE_POWER
    # prepare V and Y as needed
    V_i = V.loc[idx[1, 1:(m-1)], "V_m"] * \
        np.exp(1j*V.loc[idx[1, 1:(m-1)], "V_a"])
    V_j = V.loc[1, "V_m"] * np.exp(1j*V.loc[1, "V_a"])
    Y_ij = Y.loc[idx[1, 1:(m-1), :]].to_numpy()
    # get rid of indices for calculation
    dS = S.to_numpy() + (V_i*np.conjugate(Y_ij.dot(V_j))).to_numpy()

    # current mismatch
    dI = current_balance(V, Y, buses, NE)
    # FIXME: Find bug somewhere here, unexpected: complex f can't be separated
    # combine both
    f = np.concatenate([dS, dI])

    # error
    err_h = np.linalg.norm(f, np.Inf)
    return np.array([*f.real, *f.imag]), err_h


def harmonic_state_vector(V):
    """ returns voltages vector, magnitude then phase, without slack at h=1 """
    x = np.append(V.V_m[1:], V.V_a[1:])
    return x


def build_harmonic_jacobian(V, Y, NE):
    # preparing objects to simplify calculation
    V_vec = V.V_m*np.exp(1j*V.V_a)
    V_diag = np.diag(V_vec)
    V_norm = V_vec/V.V_m
    V_norm_diag = np.diag(V_norm)
    Y_diag = block_diag_dense(*[Y.loc[i] for i in HARMONICS])  # TODO: use sparse

    # IV and IT
    IV = Y_diag.dot(V_norm_diag)  # diagonal blocks for p = h
    IT = 1j*Y_diag.dot(V_diag)

    # iterate through YV and subtract derived current injections
    # number of harmonics (without fundamental)
    K = len(HARMONICS) - 1
    n_blocks = K+1
    nl_idx_start = list(range(m, n*(K+1), n))
    nl_V = V_vec[nl_idx_start]
    nl_V_norm = nl_V/abs(nl_V)

    for a in range(n_blocks):  # iterating through blocks (harmonics) vertically
        for b in range(n_blocks):   # ... and horizontally
            for i in range(m, n):  # iterating through nonlinear buses
                Y_N = NE[buses.loc[i].component][1]  # in NE "1" points to Y_N
                # TODO: test if this works correctly for multiple nl buses
                IV[a*n+i, b*n+i] -= Y_N.iloc[a, b]*nl_V_norm.iloc[b]
                IT[a*n+i, b*n+i] -= 1j*Y_N.iloc[a, b]*nl_V.iloc[b]
    # crop
    IV = IV[m:, 1:]
    IT = IT[m:, 1:]

    # SV and ST (from fundamental)
    # TODO: Harmonize sorting with fundamental
    V_vec_1 = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = np.diag(Y.loc[1].to_numpy().dot(V_vec_1))
    V_diag = np.diag(V_vec_1)
    V_diag_norm = np.diag(V_vec_1/abs(V_vec_1))

    S1V1 = V_diag_norm.dot(np.conj(I_diag)) \
        + V_diag.dot(np.conj(Y.loc[1].to_numpy().dot(V_diag_norm)))
    S1T1 = 1j*V_diag.dot(np.conj(I_diag - Y.loc[1].to_numpy().dot(V_diag)))

    SV = np.block([S1V1[1:m, 1:], np.zeros((m-1, n*K))])
    ST = np.block([S1T1[1:m, 1:], np.zeros((m-1, n*K))])

    J = np.block([[SV.real, ST.real],
                  [IV.real, IT.real],
                  [SV.imag, ST.imag],
                  [IV.imag, IT.imag]])
    return J


def update_harmonic_state_vec(J, x, f, sparse=SPARSE):
    """ perform Newton-Raphson iteration """
    if sparse:
        x_new = x - spsolve(J, f)
    else:
        x_new = x - spsolve(J, f)
    return x_new


def update_harmonic_voltages(V, x):
    """update and clean all voltages after iteration"""
    V.iloc[idx[1:], 0] = x[:int(len(x)/2)]
    V.iloc[idx[1:], 1] = x[int(len(x)/2):]
    # somehow this doesn't work, why? Instead only performed in the end
    # add pi to negative voltage magnitudes
    # V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    # V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change sign
    V["V_a"] = V["V_a"] % (2*np.pi)  # modulo phase wrt 2pi
    return V


def hpf(buses, lines, plt_convergence=False):
    """ execute fundamental power flow

    :param plt_convergence(default=False), shows convergence behaviour by
           plotting err_t
    :return: V: final voltages
             err_t: error over time
             n_iter_f: number of iterations performed
    """
    Y = build_admittance_matrices(buses, lines, HARMONICS)
    V, err1_t, n_converged = pf(Y, buses)
    NE = import_Norton_Equivalents(buses, COUPLED_NE)
    n_iter_h = 0
    f, err_h = harmonic_mismatch(V, Y, buses, NE)
    x = harmonic_state_vector(V)
    # import all NE of nonlinear devices present in "buses"
    err_h_t = {}
    while err_h > THRESH_H and n_iter_h < MAX_ITER_H:
        J = build_harmonic_jacobian(V, Y, NE)
        x = update_harmonic_state_vec(J, x, f)
        V = update_harmonic_voltages(V, x)
        (f, err_h) = harmonic_mismatch(V, Y, buses, NE)
        err_h_t[n_iter_h] = err_h
        n_iter_h += 1

    # getting rid of negative voltage magnitudes:
    # add pi to negative voltage magnitudes
    V.loc[V["V_m"] < 0, "V_a"] = V.loc[V["V_m"] < 0, "V_a"] - np.pi
    V.loc[V["V_m"] < 0, "V_m"] = -V.loc[V["V_m"] < 0, "V_m"]  # change signum

    # plot convergence behaviour
    if plt_convergence:
        plt.plot(list(err_h_t.keys()), list(err_h_t.values()))
    print(V)
    if n_iter_h < MAX_ITER_H:
        print("Harmonic power flow converged after " + str(n_iter_h) +
              " iterations.")
    elif n_iter_h == MAX_ITER_H:
        print("Maximum of " + str(n_iter_h) + " iterations reached.")
    return V, err_h, n_iter_h

Y = build_admittance_matrices(buses, lines, HARMONICS)
V = init_voltages(buses, HARMONICS)

(V_h, err_h_final, n_iter_h) = hpf(buses, lines)



# if __name__ == '__main__':
