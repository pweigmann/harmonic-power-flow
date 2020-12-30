# rewriting the harmonic coupled norton equivalent method in a generalized and
# modularized way

import numpy as np
import pandas as pd
from scipy.sparse.linalg import *
import matplotlib.pyplot as plt
import sys


# global variables
PU_FACTOR = 1000
HARMONICS = [1, 5, 7]
MAX_ITER_F = 30  # maybe better as argument of pf function
THRESH_F = 1e-6

# helper definitions
idx = pd.IndexSlice


# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# infrastructure (TODO: import infrastructure from file)
buses_fu = pd.DataFrame(np.array([[1, "slack", "generator", 0, 0, 1000, 0.0001],
                                  [2, "PQ", "lin_load_1", 100, 100, None, 0],
                                  [3, "PQ", None, 0, 0, None, 0],
                                  [4, "nonlinear", "nlin_load_1", 250, 100,
                                   None, 0]]),
                        columns=["ID", "type", "component", "P1",
                                 "Q1", "S1", "X_shunt"])
lines_fu = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                                  [2, 2, 3, 0.02, 0.08],
                                  [3, 3, 4, 0.01, 0.02],
                                  [4, 4, 1, 0.01, 0.02]]),
                        columns=["ID", "fromID", "toID", "R", "X"])


''' Functions for Fundamental Power Flow
'''


def build_admittance_matrices(buses, lines, harmonics):
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
    S = (buses["P1"] + 1j*buses["Q1"])/PU_FACTOR
    mismatch = np.array(V_vec*np.conj(Y.dot(V_vec)) + S, dtype="c16")
    # again following PyPSA conventions
    f = np.r_[mismatch.real[1:], mismatch.imag[1:]]
    err = np.linalg.norm(f, np.Inf)
    return f, err


def build_jacobian(V, Y):
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
    x_new = x - spsolve(J, f)  # use sparse matrices later
    return x_new


def update_fund_voltages(V, x):
    V.loc[idx[1, 1:], "V_a"] = x[:int(len(x)/2)]
    V.loc[idx[1, 1:], "V_m"] = x[int(len(x)/2):]
    return V


def pf(V, x, f, Y, buses, plt_convergence=False):
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
Y_h = build_admittance_matrices(buses_fu, lines_fu, HARMONICS)
Y_1 = np.array(Y_h.loc[1])
V_h = init_voltages(buses_fu, HARMONICS)
x_1 = init_fund_state_vec(V_h)
f_1, err1 = fund_mismatch(buses_fu, V_h, Y_1)
V_h, err1_t, n_converged = pf(V_h, x_1, f_1, Y_1, buses_fu)

if HARMONICS == [1]:
    pass
    # exit()

''' Harmonic Power Flow
n buses total (i = 1, ..., n)
slack bus is first bus (i = 1)
m-1 linear buses (i = 1, ..., m-1)
n-m+1 nonlinear buses (i = m, ..., n)
K harmonics considered (excluding fundamental)
'''

def g(v, bus):
    g = 0.3*(v.at[(1, bus), "V_m"]**3)*np.exp(3j*v.at[(1, bus), "V_p"]) +\
        0.3*(v.at[(5, bus), "V_m"]**2)*np.exp(3j*v.at[(5, bus), "V_p"])
    return g

def current_injections(busID, V, Y_N, I_N):
    # TODO: import Norton parameters from file, depending on type of device
    # busID, Y_N and I_N can all be passed/imported together
    # dimensions need to fit, crop Y_N and I_N as necessary
    V_h = V.loc[idx[:, busID], "V_m"] * np.exp(1j*V.loc[idx[:, busID], "V_a"])
    I_inj = I_N - spsolve(Y_N, V_h)
    return I_inj


def harmonic_mismatch(V, Y, buses):
    # power mismatch
    # add all linear buses to dS except slack (# = m-2)
    V_vec = 0
    dS = buses_fu.P1[buses_fu["type"] != "nonlinear"][1:]/PU_FACTOR
    + 1j*buses_fu.Q1[buses_fu["type"] != "nonlinear"][1:]/PU_FACTOR
    # current mismatch
    # at fundamental frequency for nonlinear buses
    dI_1 = Y*V_vec
    # at harmonic frequencies for all buses

    f_h = dS
    return f_h

# harmonic power flow execution

f_h = harmonic_mismatch(V_h, Y_h, buses_fu)