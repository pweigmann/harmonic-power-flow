""" Harmonic Power Flow using Norton-Raphson

rewriting the harmonic coupled norton equivalent method in a generalized and
modularized way

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
#  decide where to transform to pu system, calculation correct?
#  test for multiple nonlinear buses as well as only one nonlinear bus
#  variable naming convention

import numpy as np
import pandas as pd
from scipy.sparse.linalg import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# global variables
BASE_POWER = 1000  # could also be be imported with infra, as nominal sys power
BASE_VOLTAGE = 230
HARMONICS = [1, 5, 7]
HARMONICS_FREQ = [50 * i for i in HARMONICS]
MAX_ITER_F = 30  # maybe better as argument of pf function
THRESH_F = 1e-6  # error threshold of fundamental mismatch function
COUPLED_NE = True  # use Norton parameters of coupled vs. uncoupled model

# helper definitions
idx = pd.IndexSlice

# pu system
base_current = 1000*BASE_POWER/BASE_VOLTAGE
base_admittance = base_current/BASE_VOLTAGE

# number of harmonics (without fundamental)
K = len(HARMONICS) - 1


# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# infrastructure (TODO: import infrastructure from file)
# df for constant properties of buses
buses_const = pd.DataFrame(np.array([[1, "slack", "generator", 1000, 0.0001],
                                  [2, "PQ", "lin_load_1", None, 0],
                                  [3, "PQ", None, None, 0],
                                  [4, "nonlinear", "nlin_load_1", None, 0]]),
                           columns=["ID", "type", "component", "S", "X_shunt"])
# generate columns for all frequencies
columns = []
for h in HARMONICS:
    columns.append("P" + str(h))
    columns.append("Q" + str(h))
# df for real and reactive power of buses
buses_power = pd.DataFrame(np.zeros((len(buses_const), 2*len(HARMONICS))),
                           columns=columns)
# insert fundamental powers, part of future import
buses_power["P1"] = [0, 100, 0, 250]
buses_power["Q1"] = [0, 100, 0, 100]
# combined df for buses
buses = pd.concat([buses_const, buses_power], axis=1)
# find first nonlinear bus FIXME: start counting from 0 or 1? atm mixed
m = min(buses.index[buses["type"] == "nonlinear"])
n = len(buses)

lines_fu = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                                  [2, 2, 3, 0.02, 0.08],
                                  [3, 3, 4, 0.01, 0.02],
                                  [4, 4, 1, 0.01, 0.02]]),
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
    # TODO: import voltages from file
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


def fund_mismatch(buses, V, Y):
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    S = (buses["P1"] + 1j*buses["Q1"])/BASE_POWER
    mismatch = np.array(V_vec*np.conj(Y.dot(V_vec)) + S, dtype="c16")
    # again following PyPSA conventions
    f = np.r_[mismatch.real[1:], mismatch.imag[1:]]
    err = np.linalg.norm(f, np.Inf)
    return f, err


def build_jacobian(V, Y):
    """ fundamental Jacobian containing partial derivatives of S wrt V """
    V_vec = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    I_diag = np.diag(Y.dot(V_vec))
    V_diag = np.diag(V_vec)
    V_diag_norm = np.diag(V_vec/abs(V_vec))

    dSdt = 1j*V_diag.dot(np.conj(I_diag - Y.dot(V_diag)))

    dSdV = V_diag_norm.dot(np.conj(I_diag)) \
        + V_diag.dot(np.conj(Y.dot(V_diag_norm)))

    dPdt = dSdt[1:, 1:].real
    dPdV = dSdV[1:, 1:].real
    dQdt = dSdt[1:, 1:].imag
    dQdV = dSdV[1:, 1:].imag
    J = np.vstack([np.hstack([dPdt, dPdV]), np.hstack([dQdt, dQdV])])
    return J


def update_fund_state_vec(J, x, f):
    """ perform Newton-Raphson iteration """
    x_new = x - spsolve(J, f)  # use sparse matrices later
    return x_new


def update_fund_voltages(V, x):
    V.loc[idx[1, 1:], "V_a"] = x[:int(len(x)/2)]
    V.loc[idx[1, 1:], "V_m"] = x[int(len(x)/2):]
    return V


def pf(V, x, f, Y, buses, plt_convergence=False):
    """ execute fundamental power flow

    :param plt_convergence(default=False), shows convergence behaviour by
           plotting err_t
    :return: V: final voltages
             err_t: error over time
             n_iter_f: number of iterations performed
    """

    n_iter_f = 0
    err = np.linalg.norm(f, np.Inf)
    err_t = {}
    while err > THRESH_F and n_iter_f < MAX_ITER_F:
        J = build_jacobian(V, Y)
        x = update_fund_state_vec(J, x, f)
        V = update_fund_voltages(V, x)
        f, err = fund_mismatch(buses, V, Y)
        err_t[n_iter_f] = err
        n_iter_f += 1
    # plot convergence behaviour
    if plt_convergence:
        plt.plot(list(err_t.keys()), list(err_t.values()))
    print(V.loc[1])
    if n_iter_f < MAX_ITER_F:
        print("Converged after " + str(n_iter_f) + " iterations.")
    elif n_iter_f == MAX_ITER_F:
        print("Maximum of " + str(n_iter_f) + " iterations reached.")
    return V, err_t, n_iter_f


# fundamental power flow execution
Y_h = build_admittance_matrices(buses, lines_fu, HARMONICS)
Y_1 = np.array(Y_h.loc[1])
V_h = init_voltages(buses, HARMONICS)
x_1 = init_fund_state_vec(V_h)
f_1, err1 = fund_mismatch(buses, V_h, Y_1)
V_h, err1_t, n_converged = pf(V_h, x_1, f_1, Y_1, buses)

if HARMONICS == [1]:
    pass
    # exit()


def import_Norton_Equivalents(coupled):
    # TODO: generalize for multiple nonlinear devices
    #  import NEs from other sources or input manually
    #  chose alternative format (e.g. HDF5 or pickle instead of csv)

    # import Norton Equivalents (one type of device for now)
    # hardcoded. idea: always just this one file
    file_path = "~/Git/harmonic-power-flow/Circuit Simulation/NE.csv"
    NE_SMPS = pd.read_csv(file_path, index_col=["Parameter", "Frequency"])
    # change column type from str to int
    NE_SMPS.columns = NE_SMPS.columns.astype(int)
    # filter all columns for harmonics considered
    NE_SMPS = NE_SMPS[HARMONICS_FREQ]
    # values are imported as strings, transform to complex
    NE_SMPS = NE_SMPS.apply(lambda col: col.apply(
        lambda val: complex(val.strip('()'))))

    # change to pu system
    I_N_c = NE_SMPS.loc["I_N_c"]/base_current
    # also filter Y_N_c rows by harmonics considered
    Y_N_c = NE_SMPS.loc[("Y_N_c", HARMONICS_FREQ), HARMONICS_FREQ] / \
        base_admittance
    I_N_uc = NE_SMPS.loc["I_N_uc"]/base_current
    Y_N_uc = NE_SMPS.loc["Y_N_uc"]/base_admittance

    if coupled:
        return I_N_c, Y_N_c
    else:
        return I_N_uc, Y_N_uc


def current_injections(busID, V):
    # TODO: enhance for multiple nonlinear buses, plus automatic selection
    (I_N, Y_N) = import_Norton_Equivalents(COUPLED_NE)
    V_h = V.loc[idx[:, busID], "V_m"] * np.exp(1j*V.loc[idx[:, busID], "V_a"])
    I_inj = np.squeeze(I_N) - Y_N.dot(V_h.to_numpy()).droplevel(0)  # not ideal
    return I_inj


def current_balance(V, Y, buses):
    """ evaluate current balance at all frequencies

    Fundamental current balance only for nonlinear buses (n-m+1)
    Harmonic current balance for all buses and all harmonics (n*K)
    :return: vector of n-m+1 + nK complex current balances
    """

    # fundamental admittance for nonlinear buses
    Y_f = Y_h.loc[1, m:]
    # fundamental voltage for all buses
    V_f = V_h.loc[1, "V_m"] * np.exp(1j*V_h.loc[1, "V_a"])
    # current injections at nonlinear buses. FIXME: only works for one bus
    I_i = current_injections(3, V_h)
    # fundamental current balance
    dI_f = np.squeeze(Y_f).dot(V_f) - I_i[50]

    # construct V and Y from list of sub-arrays except fund
    # sparse matrix would be better
    Y = block_diag(*[Y_h.loc[i] for i in HARMONICS[1:]])
    V = V_h.loc[HARMONICS[1:], "V_m"] * np.exp(1j*V_h.loc[HARMONICS[1:], "V_a"])
    I_h = np.concatenate([np.array(
        [*np.zeros(n-1), i]) for i in I_i.loc[HARMONICS_FREQ[1:]]])
    dI_h = V.dot(Y) - I_h
    # final current balance vector
    dI = np.array([dI_f, *dI_h])
    return dI


def harmonic_mismatch(V, Y, buses):
    """ power and current mismatches for harmonic power flow

    also referred to as harmonic mismatch vector f_h, that needs to be minimized
    during NR algorithm
    :return: complex vector of powers (m-2) and currents (n-m+1 + nK)
    """

    # fundamental power mismatch, first iteration same as in fundamental pf: f
    # add all linear buses to S except slack (# = m-2)
    S = buses.loc[1:(m-1), "P1"]/BASE_POWER + \
        1j*buses.loc[1:(m-1), "Q1"]/BASE_POWER
    # prepare V and Y as needed
    V_i = V_h.loc[idx[1, 1:(m-1)], "V_m"] * \
        np.exp(1j*V_h.loc[idx[1, 1:(m-1)], "V_a"])
    V_j = V_h.loc[1, "V_m"] * np.exp(1j*V_h.loc[1, "V_a"])
    Y_ij = Y_h.loc[idx[1, 1:(m-1), :]].to_numpy()
    # get rid of indices for calculation
    dW = S.to_numpy() + (V_i*np.conjugate(Y_ij.dot(V_j))).to_numpy()

    # current mismatch
    dI = current_balance(V, Y, buses)

    # combine both
    f_h = np.concatenate([dW, dI])
    return f_h




def harmonic_state_vector():
    x_h = 0
    return x_h


# def build_harmonic_jacobian()







# def update_voltages()
# def hpf()

f_h = harmonic_mismatch(V_h, Y_h, buses)
I_inj = current_injections(3, V_h)

