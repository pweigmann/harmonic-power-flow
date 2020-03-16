# Doing the calculations from literature source [Almeida.2010]
# Harmonic Coupled Norton Equivalent model for a one bus system
# first index counts the harmonic, second one measurement: V_h_m. 
# Using p.u. system

import numpy as np
import pandas as pd

# harmonics 1, 3 and 5 are considered, thus 4 measurements needed:
# measurements apply different distorted voltages, measure distorted currents

rad = 2*np.pi/360  # change degree to radians

# measurement 1
V_f_1 = 1*np.exp(rad*10*1j)
V_3_1 = 0
V_5_1 = 0
# results 1
I_f_1 = 0.98*np.exp(-10*rad*1j)
I_3_1 = 0.15*np.exp(-30*rad*1j)
I_5_1 = 0.03*np.exp(-60*rad*1j)

# measurement 2
V_f_2 = 0.95
V_3_2 = 0.03*np.exp(10*rad*1j)
V_5_2 = 0.01*np.exp(30*rad*1j)
# results 2
I_f_2 = 1*np.exp(-5*rad*1j)
I_3_2 = 0.25*np.exp(-60*rad*1j)
I_5_2 = 0.05*np.exp(-55*rad*1j)

# measurement 3
V_f_3 = 1.05*np.exp(1*rad*1j)
V_3_3 = 0.03*np.exp(10*rad*1j)
V_5_3 = 0.005*np.exp(90*rad*1j)
# results 3
I_f_3 = 0.75*np.exp(-15*rad*1j)
I_3_3 = 0.25*np.exp(-35*rad*1j)
I_5_3 = 0.05*np.exp(-75*rad*1j)

# measurement 4
V_f_4 = 1.10*np.exp(3*rad*1j)
V_3_4 = 0.05*np.exp(30*rad*1j)
V_5_4 = 0.01*np.exp(55*rad*1j)
# results 4
I_f_4 = 0.95*np.exp(-5*rad*1j)
I_3_4 = 0.35*np.exp(-10*rad*1j)
I_5_4 = 0.15*np.exp(-30*rad*1j)

# voltages matrix
V = np.array([
    [V_f_1, V_3_1, V_5_1, 1],
    [V_f_2, V_3_2, V_5_2, 1],
    [V_f_3, V_3_3, V_5_3, 1],
    [V_f_4, V_3_4, V_5_4, 1]
])

# currents vector, fundamental frequency
I = np.array([I_f_1, I_f_2, I_f_3, I_f_4])

# invert voltages matrix
V_inv = np.linalg.inv(V)

# solve for Norton admittances and current source, fundamental frequency
Y_I_Norton_f = V_inv.dot(I)  # NOT the same results as in the paper...


