# Minimal, not-at-all-automated, 3 Bus example of Norton Equivalent Model
# for Harmonic Power Flow based on PyPSA "pypsa_minimal_pf.py".

# Author: Pascal Weigmann, p.weigmann@posteo.de

import numpy as np
import pandas as pd
from scipy.sparse.linalg import *

pu_factor = 400  # as in PyPSA example


# buses and there initial state
bus0 = {"type": "slack", "P": 100, "Q": 0, "V": 1, "theta": 0}  # gen, slack bus
bus1 = {"type": "PQ", "P": -80, "Q": -80, "V": 1, "theta": 0}  # load
bus2 = {"type": "PQ", "P": -20, "Q": -20, "V": 1, "theta": 0}  # nonlinear load

# lines (no shunt impedances)
line0 = {"from": "bus0", "to": "bus1", "x": 0.1, "R": 0.01}
line1 = {"from": "bus1", "to": "bus2", "x": 0.1, "R": 0.01}
line2 = {"from": "bus2", "to": "bus0", "x": 0.1, "R": 0.01}

# line admittances
y01 = 1/(line0["R"]/pu_factor + 1j*line0["x"]/pu_factor)
y12 = 1/(line1["R"]/pu_factor + 1j*line1["x"]/pu_factor)
y20 = 1/(line2["R"]/pu_factor + 1j*line2["x"]/pu_factor)

# grid admittance matrix, fundamental frequency
Y_f = np.array([[y01+y20, -y01, -y20],
                [-y01, y12+y01, -y12],
                [-y12, -y20, y12+y20]])

n_iter = 0
max_iter = 20
err = 1
rad = 2*np.pi/360  # change degree to radians

while err > 10e-2 and n_iter < max_iter:
    # count iterations
    n_iter += 1

    # state vector x with unknown U and theta at PQ and PV buses
    x = np.array([bus1["theta"], bus2["theta"], bus1["V"], bus2["V"]])

    # mismatch function of which to find the zero using NR
    V = np.array([bus0["V"]*np.exp(1j*bus0["theta"]),
                  bus1["V"]*np.exp(1j*bus1["theta"]),
                  bus2["V"]*np.exp(1j*bus2["theta"])])
    S = np.array([bus0["P"] + 1j*bus0["Q"],
                  bus1["P"] + 1j*bus1["Q"],
                  bus2["P"] + 1j*bus2["Q"]])
    V_diag = np.array([[V[0], 0, 0], [0, V[1], 0], [0, 0, V[2]]])
    V_diag_norm = np.array([[V[0], 0, 0], [0, V[1], 0], [0, 0, V[2]]])/abs(V)
    mismatch = V*np.conj(Y_f.dot(V)) - S

    # final mismatch function as used in NR
    f = np.r_[mismatch.real[1:], mismatch.imag[1:]]

    # error to be minimized
    err = np.linalg.norm(f, np.Inf)

    # calculate Jacobian (close to PyPSA example)
    # delta S/delta theta matrix
    dSdt = 1j*V_diag.dot(np.conj(Y_f.dot(V) - Y_f.dot(V_diag)))

    # delta S/delta V matrix (not yet calculated by hand)
    dSdV = V_diag_norm.dot(np.conj(Y_f.dot(V))) + \
           V_diag.dot(np.conj(Y_f.dot(V_diag_norm)))
    J00 = dSdt[1:, 1:].real
    J01 = dSdV[1:, 1:].real
    J10 = dSdt[1:, 1:].imag
    J11 = dSdV[1:, 1:].imag

    # build final Jacobian
    J = np.vstack([np.hstack([J00, J01]), np.hstack([J10, J11])])

    # update state vector
    x2 = x - spsolve(J, f)  # can be used for sparse matrices later

    # update buses (bus0 is slack and constant)
    bus1["theta"] = x2[0]
    bus2["theta"] = x2[1]
    bus1["V"] = x2[2]
    bus2["V"] = x2[3]

if n_iter == max_iter:
    print("Maximum number of iterations reached.")
else:
    print("Converged after %s iterations" % n_iter)

print("Final voltages at buses: \n bus0: %s \n bus1: %s \n bus2: %s"
      % (bus0["V"], bus1["V"], bus2["V"]))

# Harmonic power flow with nonlinear load at bus 2

# extend buses, initial guess for harmonic voltages, common approach: V = 0.1 pu
bus0.update({"V3": 0.1, "V5": 0.1, "theta3": 0, "theta5": 0})
bus1.update({"V3": 0.1, "V5": 0.1, "theta3": 0, "theta5": 0})
bus2.update({"V3": 0.1, "V5": 0.1, "theta3": 0, "theta5": 0})

# line impedance scaled linearly (Fuchs)
line0.update({"x3": line0["x"]*3, "x5": line0["x"]*5})
line1.update({"x3": line1["x"]*3, "x5": line1["x"]*5})
line2.update({"x3": line2["x"]*3, "x5": line2["x"]*5})

# line admittances, harmonic extension
y01_3 = 1/(line0["R"]/pu_factor + 1j*line0["x3"]/pu_factor)
y12_3 = 1/(line1["R"]/pu_factor + 1j*line1["x3"]/pu_factor)
y20_3 = 1/(line2["R"]/pu_factor + 1j*line2["x3"]/pu_factor)

y01_5 = 1/(line0["R"]/pu_factor + 1j*line0["x5"]/pu_factor)
y12_5 = 1/(line1["R"]/pu_factor + 1j*line1["x5"]/pu_factor)
y20_5 = 1/(line2["R"]/pu_factor + 1j*line2["x5"]/pu_factor)

# grid admittance matrix, harmonic frequencies, scaled linearly (probably wrong)
#Y_3 = np.array([[y01+y20, -y01, -y20],
#                [-y01, y12+y01, -y12],
#                [-y12, -y20, y12+y20]])/3

# grid admittance matrix, harmonic frequencies using harmonic line impedances
# do I need to add shunt admittances for linear buses at harmonic frequencies?
Y_3 = np.array([[y01_3+y20_3, -y01_3, -y20_3],
                [-y01_3, y12_3+y01_3, -y12_3],
                [-y12_3, -y20_3, y12_3+y20_3]])
Y_5 = np.array([[y01_5+y20_5, -y01_5, -y20_5],
                [-y01_5, y12_5+y01_5, -y12_5],
                [-y12_5, -y20_5, y12_5+y20_5]])

# voltage vectors at harmonic frequencies
V3 = np.array([bus0["V3"] * np.exp(1j * bus0["theta3"]),
               bus1["V3"] * np.exp(1j * bus1["theta3"]),
               bus2["V3"] * np.exp(1j * bus2["theta3"])])
V5 = np.array([bus0["V5"] * np.exp(1j * bus0["theta5"]),
               bus1["V5"] * np.exp(1j * bus1["theta5"]),
               bus2["V5"] * np.exp(1j * bus2["theta5"])])


# Norton parameter, nonlinear load at bus2 (from almeida)
Y_N = np.array([
    [-0.79-0.981j, 6.065+8.387j, -38.4-25.34j],
    [-1.216-0.982j, -1.068+5.375j, -2.724-5.45j],
    [-0.649+0.276j, 1.858+2.038j, -9.886+0.956j]
])

I_N = np.array([
    1.165*np.exp(-81.34*rad*1j),
    1.515*np.exp(-135.72*rad*1j),
    0.682*np.exp(158.49*rad*1j)
])

bus2.update({"I_N": I_N, "Y_N": Y_N})


# function to calculate current injections based on voltage for bus2
def current_inj(V):
    I_inj = I_N - Y_N.dot(V)
    return I_inj


I_inj = current_inj(np.array([bus2['V'], bus2['V3'], bus2['V5']]))

# update vector
# with fundamental voltages from power flow and initial harmonic values
U = np.array([bus1["theta"], bus1["V"], bus2["theta"], bus2["V"],
              bus0["theta3"], bus0["V3"], bus1["theta3"], bus1["V3"],
              bus2["theta3"], bus2["V3"], bus0["theta5"], bus0["V5"],
              bus1["theta5"], bus1["V5"], bus2["theta5"], bus2["V5"],
              ])

# current mismatch for nonlinear buses
# (fuchs divides all of these into real and imaginary parts)
I_lines_f = Y_f.dot(V)
I_lines_3 = Y_3.dot(V3)
I_lines_5 = Y_5.dot(V5)

dI_bus2_f = I_inj[0] + I_lines_f[1] + I_lines_f[2]
dI_bus2_3 = I_inj[1] + I_lines_3[2] + I_lines_3[1]
dI_bus2_5 = I_inj[2] + I_lines_5[2] + I_lines_5[1]

# harmonic current mismatch for all buses
dI_bus0_3 = I_lines_3[0] + I_lines_3[2]
dI_bus0_5 = I_lines_5[0] + I_lines_5[2]
dI_bus1_3 = I_lines_3[0] + I_lines_3[1]
dI_bus1_5 = I_lines_5[0] + I_lines_5[1]
# these are all zero initially, maybe wrong?

# power mismatch (calculate power by currents, compare to given power)
I_inj_f_abs = 0  # magnitute
I_inj_f_delta = 0  # phase
P_bus2_f = I_inj_f_abs*bus2["V"]*np.cos(I_inj_f_delta - bus2["theta"])
