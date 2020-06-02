# rewriting the harmonic coupled norton equivalent method in a generalized and
# modularized way

import numpy as np
import pandas as pd

# global variables
PU_FACTOR = 1000
HARMONICS = [1, 5, 7, 11]

# infrastructure (TODO: import infrastructure from file)
buses = pd.DataFrame(np.array([[1, "slack", "generator", 0, 0, 1000, 0.0001],
                               [2, "PQ", "lin_load_1", 100, 100, None, 0],
                               [3, "PQ", None, 0, 0, None, 0],
                               [4, "nonlinear", "nlin_load_1", 250, 100, None,
                                0]]),
                     columns=["ID", "type", "component", "P1",
                              "Q1", "S1", "y_shunt"])
lines = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                               [2, 2, 3, 0.02, 0.08],
                               [3, 3, 4, 0.01, 0.02],
                               [4, 4, 5, 0.01, 0.02]]),
                     columns=["ID", "fromID", "toID", "R", "X"])


def build_admittance_matrix(buses, lines):
    Y = np.zeros([len(buses), len(buses)], dtype=complex)
    # non-diagonal elements
    for idx, line in lines.iterrows():
        Y[int(line.fromID - 1), int(line.toID - 1)] = -1/(line.R + 1j*line.X)
        # admittance matrix is assumed to be symmetric
        Y[int(line.toID - 1), int(line.fromID - 1)] = \
            Y[int(line.fromID - 1), int(line.toID - 1)]
    # diagonal elements
    for m in range(len(buses)):
        Y[m, m] = buses.y_shunt[m] - sum(Y[m, ])
    return Y


def init_voltages(buses, harmonics):
    iterables = [harmonics, buses.index.values]
    multi_idx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
    # TODO: import voltages from file
    V = pd.DataFrame(np.zeros((len(harmonics) * len(buses), 2)),
                     index=multi_idx, columns=["V_m", "V_a"])
    V.sort_index(inplace=True)
    # set initial voltage magnitudes (in p.u.)
    V.loc[1, "V_m"] = 1
    V.loc[harmonics[1]:, "V_m"] = 0.1
    return V


def init_fund_state_vec(V):
    # following pypsa instead of Fuchs by not alternating between angle and magn
    x = np.array(V.loc[(1, "V_a")][1:], V.loc[(1, "V_m")][1:])
    return x


def init_fund_mismatch(V):
    f =
    return f