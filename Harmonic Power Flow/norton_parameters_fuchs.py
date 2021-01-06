# Calculating the Norton Parameters that correspond to the calculations
# performed by Fuchs in his example.

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

# harmonics 1, 5 are considered, thus 3 measurements needed:
# measurements apply different distorted voltages, measure distorted currents

rad = 2*np.pi/360  # change degree to radians

# calculation of the current injections at bus 4
iterables = [[1, 5], ["bus1", "bus2", "bus3", "bus4"]]
multiIdx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
V = pd.DataFrame(np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
                 index=multiIdx, columns=["V_m", "V_a"])
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
    G_5 = abs(g5(V, "bus4"))*np.exp(1j*(
            V.at[(5, "bus4"), "V_a"] -
            np.arctan(abs(np.imag(g5(V, "bus4")))/abs(np.real(g5(V, "bus4"))))))
    return G_1, G_5


# measurement 1
V1 = copy.deepcopy(V)
V1.loc[(slice(None), "bus4"), :] = [[1, 0], [0.1, 0]]
# results 1
I_f_1 = inj(V1)[0]
I_5_1 = inj(V1)[1]

# measurement 2
V2 = copy.deepcopy(V)
V2.loc[(slice(None), "bus4"), :] = [[1, 0], [0.1, 0.1]]
# results 1
I_f_2 = inj(V2)[0]
I_5_2 = inj(V2)[1]

# measurement 3
V3 = copy.deepcopy(V)
V3.loc[(slice(None), "bus4"), :] = [[0.9, 0.1], [0.2, 0]]
# results 1
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