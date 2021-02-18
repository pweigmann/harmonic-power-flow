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
V_f_1 = 1
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

# voltages matrix (corrected signum mistake in Almeida paper, eq. 8!)
V = -np.array([
    [V_f_1, V_3_1, V_5_1, -1],
    [V_f_2, V_3_2, V_5_2, -1],
    [V_f_3, V_3_3, V_5_3, -1],
    [V_f_4, V_3_4, V_5_4, -1]
])

# currents vector, fundamental frequency
I_f = np.array([I_f_1, I_f_2, I_f_3, I_f_4])

# invert voltages matrix
V_inv = np.linalg.inv(V)

# solve for Norton admittances and current source, fundamental frequency
Y_I_Norton_f = V_inv.dot(I_f)  # with correction, same results

# let's try the next step:
# calculate harmonics with Norton parameters as mentioned in the paper
Y_N_paper = np.array([
    [-0.79-0.981j, 6.065+8.387j, -38.4-25.34j],
    [-1.216-0.982j, -1.068+5.375j, -2.724-5.45j],
    [-0.649+0.276j, 1.858+2.038j, -9.886+0.956j]
])

I_N_paper = np.array([
    1.165*np.exp(-81.34*rad*1j),
    1.515*np.exp(-135.72*rad*1j),
    0.682*np.exp(158.49*rad*1j)
])

# test if these NE return original results when original voltage applied
I_test_m1 = I_N_paper - Y_N_paper.dot(np.array([1, 0, 0]))
I_test_m2 = I_N_paper - Y_N_paper.dot(np.array([V_f_2, V_3_2, V_5_2]))
I_test_m3 = I_N_paper - Y_N_paper.dot(np.array([V_f_3, V_3_3, V_5_3]))
I_test_m4 = I_N_paper - Y_N_paper.dot(np.array([V_f_4, V_3_4, V_5_4]))
# --> they do, correct NEs

# line impedance and admittance (scaled with frequency...?)
# (now scaled, so THD_v matched the paper)
Z_Line_f = 0.05 + 0.25j
Y_Line_f = 1/Z_Line_f

Z_Line_3 = Z_Line_f*1.5
Y_Line_3 = 1/Z_Line_3

Z_Line_5 = Z_Line_f*2
Y_Line_5 = 1/Z_Line_5

# defining sub-matrices
Y_ss = np.array([
    [Y_Line_f, 0, 0],
    [0, Y_Line_3, 0],
    [0, 0, Y_Line_5]
])

Y_sl = np.array([
    [-Y_Line_f, 0, 0],
    [0, -Y_Line_3, 0],
    [0, 0, -Y_Line_5]
])

Y_ls = Y_sl

Y_ll = Y_N_paper + Y_ss

Y_ll_inv = np.linalg.inv(Y_ll)

# given are supply voltages and load currents
V_s_I_l = np.concatenate((np.array([V_f_3, V_3_3, V_5_3]), I_N_paper))

# calculating unknown voltages and currents
Y = np.concatenate((np.concatenate((Y_ss, Y_sl), axis=1),
                    np.concatenate((Y_ls, Y_ll), axis=1)))
# (see Almeida.2010)
I_s_V_l = np.concatenate((np.concatenate((Y_ss - (Y_sl.dot(Y_ll_inv).dot(Y_ls)),
                                          Y_sl.dot(Y_ll_inv)), axis=1),
                          np.concatenate((-Y_ll_inv.dot(Y_ls), Y_ll_inv),
                                         axis=1))).dot(V_s_I_l)

# calculating THD of voltage
# (correct? no -> change to polar coordinates to get correct amplitude)
THD_v = np.sqrt(I_s_V_l[4]**2 + I_s_V_l[5]**2)/I_s_V_l[3]
print(THD_v)
