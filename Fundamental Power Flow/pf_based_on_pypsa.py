# Minimal, not-at-all-automated, 3 Bus example of Norton Equivalent Model
# for Harmonic Power Flow based on PyPSA "pypsa_minimal_pf.py".

# Author: Pascal Weigmann, p.weigmann@posteo.de

import numpy as np
from scipy.sparse.linalg import *

pu_factor = 400  # as in PyPSA example

# buses and there initial state
bus0 = {"type": "slack", "P": 100, "Q": 0, "V": 1, "theta": 0}  # gen, slack bus
bus1 = {"type": "PQ", "P": -100, "Q": -100, "V": 1, "theta": 0}  # load
bus2 = {"type": "PQ", "P": 0, "Q": 0, "V": 1, "theta": 0}

# lines (no shunt impedances)
line0 = {"from": "bus0", "to": "bus1", "x": 0.1, "R": 0.01}
line1 = {"from": "bus1", "to": "bus2", "x": 0.1, "R": 0.01}
line2 = {"from": "bus2", "to": "bus0", "x": 0.1, "R": 0.01}

# admittance matrix, fundamental frequency
y01 = 1/(line0["R"]/pu_factor + 1j*line0["x"]/pu_factor)
y12 = 1/(line1["R"]/pu_factor + 1j*line1["x"]/pu_factor)
y20 = 1/(line2["R"]/pu_factor + 1j*line2["x"]/pu_factor)

Y_f = np.array([[y01+y20, -y01, -y20],
                [-y01, y12+y01, -y12],
                [-y12, -y20, y12+y20]])

n_iter = 0
max_iter = 20
err = 1

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
