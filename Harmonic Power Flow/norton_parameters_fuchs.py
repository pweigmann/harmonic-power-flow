# Calculating the Norton Parameters that correspond to the calculations
# performed by Fuchs in his example.

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# harmonics 1, 5 are considered, thus 3 measurements needed:
# measurements apply different distorted voltages, measure distorted currents

rad = 2*np.pi/360  # change degree to radians

# calculation of the current injections at bus 4
iterables = [[1, 5], ["bus1", "bus2", "bus3", "bus4"]]
multiIdx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
V = pd.DataFrame(np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
                 index=multiIdx, columns=["V_m", "V_a"])

# import logged voltages from "example_hpf_fuchs"
V_log = pd.read_json("V_log.json", orient="table")
# I_log = pd.read_json("I_log.json", orient="table") some problem with import

# slice V to obtain the voltages needed for the virtual measurement at bus4
V_mes = V_log.loc[pd.IndexSlice[0:2, :, "bus4"], :]

# real and reactive power at fund frequency are constant
P_1 = 0.25
Q_1 = 0.1
eps_1 = np.arctan(Q_1/P_1)


def g5(V, bus):
    g5 = 0.3*(V.at[(1, bus), "V_m"]**3)*np.exp(3j*V.at[(1, bus), "V_a"]) +\
         0.3*(V.loc[(5, bus), "V_m"]**2)*np.exp(3j*V.at[(5, bus), "V_a"])
    return g5


def inj(V):
    G_1 = P_1*np.exp(1j*(V.at[(1, "bus4"), "V_a"]-eps_1))\
          /V.at[(1, "bus4"), "V_m"]*np.cos(eps_1)
    G_5 = g5(V, "bus4")
    # as described in the book
    # G_5 = abs(g5(V, "bus4"))*np.exp(1j*(V.at[(5, "bus4"), "V_a"] -
    #        np.arctan(abs(np.imag(g5(V, "bus4")))/
    #        abs(np.real(g5(V, "bus4"))))))
    return G_1, G_5


# measurement 1, from simulation voltages
V1 = V_mes.loc[0]

# results 1
I_f_1 = inj(V1)[0]
I_5_1 = inj(V1)[1]

# measurement 2, from simulation voltages
V2 = V_mes.loc[1]

# results 2
I_f_2 = inj(V2)[0]
I_5_2 = inj(V2)[1]

# measurement 3, from simulation voltages
V3 = V_mes.loc[2]

# results 3
I_f_3 = inj(V3)[0]
I_5_3 = inj(V3)[1]


# voltages matrix
V_m = np.array([
    [V1.loc[(1, "bus4"), "V_m"]*np.exp(1j*V1.loc[(1, "bus4"), "V_a"]),
     V1.loc[(5, "bus4"), "V_m"]*np.exp(1j*V1.loc[(5, "bus4"), "V_a"]), 1],
    [V2.loc[(1, "bus4"), "V_m"]*np.exp(1j*V2.loc[(1, "bus4"), "V_a"]),
     V2.loc[(5, "bus4"), "V_m"]*np.exp(1j*V2.loc[(5, "bus4"), "V_a"]), 1],
    [V3.loc[(1, "bus4"), "V_m"]*np.exp(1j*V3.loc[(1, "bus4"), "V_a"]),
     V3.loc[(5, "bus4"), "V_m"]*np.exp(1j*V3.loc[(5, "bus4"), "V_a"]), 1],
])

# currents vector, fundamental frequency
I_f = np.array([I_f_1, I_f_2, I_f_3])
I_5 = np.array([I_5_1, I_5_2, I_5_3])

# invert voltages matrix
V_m_inv = np.linalg.inv(V_m)

# solve for Norton admittances and current source
Y_I_Norton_f = V_m_inv.dot(I_f)
Y_I_Norton_5 = V_m_inv.dot(I_5)

# build Norton parameters
Y_N = np.array([[Y_I_Norton_f[0], Y_I_Norton_f[1]],
                [Y_I_Norton_5[0], Y_I_Norton_5[1]]])

I_N = np.array([Y_I_Norton_f[2], Y_I_Norton_5[2]])

print(Y_N)
print(I_N)

XY = Y_N.real
YY = Y_N.imag

XI = I_N.real
YI = I_N.imag
plt.scatter(XY, YY, c="red")
plt.scatter(XI, YI, c="blue")
plt.show()


# test
# voltage measurements as vectors
V1_vec = np.squeeze(np.array([V1.loc[1, "V_m"]*np.exp(1j*V1.loc[1, "V_a"]),
                              V1.loc[5, "V_m"]*np.exp(1j*V1.loc[5, "V_a"])]))
V2_vec = np.squeeze(np.array([V2.loc[1, "V_m"]*np.exp(1j*V2.loc[1, "V_a"]),
                              V2.loc[5, "V_m"]*np.exp(1j*V2.loc[5, "V_a"])]))
V3_vec = np.squeeze(np.array([V3.loc[1, "V_m"]*np.exp(1j*V3.loc[1, "V_a"]),
                              V3.loc[5, "V_m"]*np.exp(1j*V3.loc[5, "V_a"])]))

# calculating injections with Norton parameters
I1_inj_test = I_N - Y_N.dot(V1_vec)
I2_inj_test = I_N - Y_N.dot(V2_vec)
I3_inj_test = I_N - Y_N.dot(V3_vec)
I_f_test = np.array([I1_inj_test[0], I2_inj_test[0], I3_inj_test[0]])
I_5_test = np.array([I1_inj_test[1], I2_inj_test[1], I3_inj_test[1]])

print(I_f - I_f_test)
print(I_5 - I_5_test)
