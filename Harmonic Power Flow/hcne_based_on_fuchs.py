# Copy paste of 4 Bus example of "classical" harmonic load power based on Fuchs.
# Experiment to cut away "control parameters" alpha and beta for nonlinear
# buses thus shortening dM, J_1, J_5 and U.

# TODO:
#  remove unnecessary parts
#  what to do about phasor calculations? 

# Author: Pascal Weigmann, p.weigmann@posteo.de


import numpy as np
import pandas as pd

pu_factor = 1000


# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(x):
    return abs(x), np.angle(x)


# initialize voltage as multi-index DataFrame, default: V = 1pu, V_h = 0.1pu
iterables = [[1, 5], ["bus1", "bus2", "bus3", "bus4"]]
multiIdx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
V = pd.DataFrame(np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
                 index=multiIdx, columns=["V_m", "V_p"])
V.sort_index(inplace=True)

# grid: buses and lines with their constant properties
buses = pd.DataFrame(np.array([[1, "slack", 0, 0, 1000, 0.0001],
                               [2, "PQ", 100, 100, None, 0],
                               [3, "PQ", 0, 0, None, 0],
                               [4, "nonlinear", 250, 100, None, 0]]),
                     columns=["ID", "type", "P", "Q", "S", "X_shunt"])
lines = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                               [2, 2, 3, 0.02, 0.08],
                               [3, 3, 4, 0.01, 0.02],
                               [4, 4, 1, 0.01, 0.02]]),
                     columns=["ID", "fromID", "toID", "R", "X"])  # add dtype?

# 7.3.5: construct Y_f
Y_f = np.zeros([len(buses), len(buses)], dtype=complex)
# non-diagonal elements
for k in range(0, lines.shape[0]):
    Y_f[int(lines.fromID[k])-1, int(lines.toID[k])-1] = \
        -1/(lines.R[k] + 1j*lines.X[k])
    Y_f[int(lines.toID[k])-1, int(lines.fromID[k])-1] =\
        Y_f[int(lines.fromID[k])-1, int(lines.toID[k])-1]

# diagonal elements
for n in range(0, len(buses)):
    for m in range(0, len(buses)):
        if n == m:
            # FIXME: won't work if there are more than two lines per bus
            #  simpler to use sum of values calculated above
            Y_f[n, m] = 1/(lines[lines.fromID == n+1].R.iloc[0] +
                           1j*lines[lines.fromID == n+1].X.iloc[0]) +\
                        (1/(lines[lines.toID == n+1].R.iloc[0] +
                            1j*lines[lines.toID == n+1].X.iloc[0]))
# change to polar base for easier comparison to Fuchs example
Y_f_p = np.zeros([len(buses), len(buses)], dtype=tuple)
for n in range(0, len(buses)):
    for m in range(0, len(buses)):
        Y_f_p[n, m] = A2P(Y_f[n, m])

# --> correct!

# 7.3.6: fundamental mismatch vector (doesn't contain slack bus)
V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_p"])  # build complex voltage vector
V_init = pd.Series([])
# arrangement following Fuchs' standard
for n in range(0, len(V_f)):
    V_init = V_init.append(pd.Series([V_f[n].imag, V_f[n].real],
                                     index=["V_" + str(n) + "_theta",
                                     "V_" + str(n) + "_m"]))
S = buses["P"]/pu_factor+1j*buses["Q"]/pu_factor  # apparent power
dm = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)  # power mismatch

# Newton Raphson variables (without slack bus)
x = V_init[2:]
f = np.zeros(2*(len(dm)-1))
for n in range(1, len(dm)):
    f[2*n-2] = dm[n].real
    f[2*n-1] = dm[n].imag

err = np.linalg.norm(dm, np.Inf)  # infinity norm best choice?

# 7.3.7: fundamental Jacobian (calculated the way pypsa does)
I_diag = np.diag(Y_f.dot(V_f))
V_diag = np.diag(V_f)
V_diag_norm = np.diag(V_f/abs(V_f))

dSdt = 1j*V_diag.dot(np.conj(I_diag - Y_f.dot(V_diag)))
dPdt = dSdt.real
dQdt = dSdt.imag

dSdV = V_diag_norm.dot(np.conj(I_diag)) + \
       V_diag.dot(np.conj(Y_f.dot(V_diag_norm)))
dPdV = dSdV.real
dQdV = dSdV.imag

# order of jacobian entries differs from pypsa to fuchs, sorting:
Jb = np.zeros((2*len(buses), 2*len(buses)))
for n in range(len(buses)):
    for m in range(len(buses)):
        Jb[2*n, 2*m] = dPdt[n, m]
        Jb[2*n+1, 2*m] = dQdt[n, m]
        Jb[2*n, 2*m+1] = dPdV[n, m]
        Jb[2*n+1, 2*m+1] = dQdV[n, m]

J = Jb[2:, 2:]  # correct! (Fuchs, p.591)

# 7.3.8: calculate the inverse of the Jacobian
J_inv = np.linalg.inv(J)

# 7.3.10: compute correction vector + NR iteration
x_new = x - J_inv.dot(f)


n_iter = 0
# with new V: calculate again f, J  (loop probably shouldn't start here, FIXME)
while err > 1e-6 and n_iter < 100:
    n_iter += 1
    V.loc[1, 'V_p'][1:] = x_new[::2]
    V.loc[1, 'V_m'][1:] = x_new[1::2]
    V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_p"])
    dm = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)  # recalculate dm
    x = x_new  # update x
    for n in range(1, len(dm)):  # update f
        f[2*n-2] = dm[n].real
        f[2*n-1] = dm[n].imag

    err = np.linalg.norm(f, np.Inf)

    # recalculate J
    I_diag = np.diag(Y_f.dot(V_f))
    V_diag = np.diag(V_f)
    V_diag_norm = np.diag(V_f/abs(V_f))
    dSdt = 1j*V_diag.dot(np.conj(I_diag - Y_f.dot(V_diag)))
    dPdt = dSdt.real
    dQdt = dSdt.imag

    dSdV = V_diag_norm.dot(np.conj(I_diag)) + \
           V_diag.dot(np.conj(Y_f.dot(V_diag_norm)))
    dPdV = dSdV.real
    dQdV = dSdV.imag

    for n in range(len(buses)):
        for m in range(len(buses)):
            Jb[2*n, 2*m] = dPdt[n, m]
            Jb[2*n+1, 2*m] = dQdt[n, m]
            Jb[2*n, 2*m+1] = dPdV[n, m]
            Jb[2*n+1, 2*m+1] = dQdV[n, m]

    J = Jb[2:, 2:]
    J_inv = np.linalg.inv(J)
    x_new = x - J_inv.dot(f)
    print("error: " + str(err))
print(x_new)  # voltage magnitudes same as fuchs, angles a bit off, why?
print(str(n_iter) + " iterations")

# HARMONIC POWER FLOW, only 5th harmonic considered
h = 5

# 7.4.7: Computation of harmonic admittance matrix
# Swing bus has different representation - fundamental vs. harmonic (p. 288)
# Harmonic admittance matrix, as fund. but reactance scales with harmonic no.
Y_5 = np.zeros([len(buses), len(buses)], dtype=complex)
# non-diagonal elements
for k in range(0, len(lines)):
    Y_5[int(lines.fromID[k])-1, int(lines.toID[k])-1] = \
        -1/(lines.R[k] + 1j*lines.X[k]*h)
    Y_5[int(lines.toID[k])-1, int(lines.fromID[k])-1] =\
        Y_5[int(lines.fromID[k])-1, int(lines.toID[k])-1]
# diagonal elements
# slack self admittance added as subtransient(?) admittance (p.288/595)
for n in range(len(buses)):
    if buses["X_shunt"][n] == 0:
        Y_5[n, n] = -sum(Y_5[n, :])
    else:
        Y_5[n, n] = -sum(Y_5[n, :]) + 1/(1j*buses["X_shunt"][n]*h)
# --> correct


# 7.4.8: Computation of nonlinear load harmonic currents
# Fuchs' non-linear device parameters alpha and beta are zero in this example
# TODO: this will be replaced by Norton Equivalent calculations,
def g(v, bus):
    g = 0.3*(v.at[(1, bus), "V_m"]**3)*np.exp(3j*v.at[(1, bus), "V_p"]) +\
        0.3*(v.at[(5, bus), "V_m"]**2)*np.exp(3j*v.at[(5, bus), "V_p"])
    return g

buses.rename({"P": "P_1", "Q": "Q_1"}, axis=1, inplace=True)
buses = buses.assign(P_5=np.zeros(len(buses)), Q_5=np.zeros(len(buses)))

# start iteration here?
# U as on p.280, eq. (7-93)
UV = np.hstack(np.split(V[["V_p", "V_m"]], len(V)))  # create U by rearranging V
U = UV[0, 2:]  # without control angles, cut off V_f of slack

# everything only for bus4
# this whole part might not be needed when not using alpha, beta
epsilon_1 = np.arctan(buses.at[(3, "Q_1")]/buses.at[(3, "P_1")])
gamma_1 = V.at[(1, "bus4"), "V_p"] - epsilon_1  # current phase
# fundamental "device currents" (why different to injection g(?))
G_bus4_1_r = buses.at[(3, "P_1")]/pu_factor * np.cos(gamma_1) / \
             (V.at[(1, "bus4"), "V_m"] *
              np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1))
G_bus4_1_i = buses.at[(3, "P_1")]/pu_factor * np.sin(gamma_1) / \
             (V.at[(1, "bus4"), "V_m"] *
              np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1))
G_bus4_1 = G_bus4_1_r + 1j*G_bus4_1_i

# now for h = 5
G_bus4_5 = g(V, "bus4")  # difference between G and g?
# answer: g refers to bus4, G refers to swing bus
epsilon_5 = np.arctan(abs(G_bus4_5.imag)/abs(G_bus4_5.real))
gamma_5 = V.at[(5, "bus4"), "V_p"] - epsilon_5
G_bus4_5_r = abs(G_bus4_5)*np.cos(gamma_5)
G_bus4_5_i = abs(G_bus4_5)*np.sin(gamma_5)
# --> correct (except rounding and phase sign)

# test
P_4_1 = abs(G_bus4_1)*V.at[(1, "bus4"), "V_m"] * \
        np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1)
Q_4_1 = abs(G_bus4_1)*V.at[(1, "bus4"), "V_m"] * \
        np.sin(V.at[(1, "bus4"), "V_p"] - gamma_1)

P_4_5 = abs(G_bus4_5)*V.at[(5, "bus4"), "V_m"] * \
        np.cos(V.at[(5, "bus4"), "V_p"] - gamma_5)
Q_4_5 = abs(G_bus4_5)*V.at[(5, "bus4"), "V_m"] * \
        np.sin(V.at[(5, "bus4"), "V_p"] - gamma_5)

# total injected powers at bus4
P_4_t = P_4_1 + P_4_5
Q_4_t = Q_4_1 + Q_4_5

# 7.4.9: Evaluation of harmonic mismatch vector dM = [dW, dI_5, dI_1]
# dW for the linear buses (analog to dm in 7.3.6):
V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_p"])
dW_lin = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)
# --> almost zero, which makes sense because it was minimized during fund. pf
# but Fuchs has bigger values (rounding?)

# dW for nonlinear buses needs harmonic line powers
V_5 = V.loc[5, "V_m"]*np.exp(1j*V.loc[5, "V_p"])
F_nlin = np.array(V_f*np.conj(Y_f.dot(V_f))) + \
         np.array(V_5*np.conj(Y_5.dot(V_5)))
dW_nlin = F_nlin[3] + P_4_t + 1j*Q_4_t  # also different to Fuchs
# final dW (excluding dW_nlin eq. (7-100) on p. 281)
dW = np.array([dW_lin[1].real, dW_lin[1].imag, dW_lin[2].real, dW_lin[2].imag])

# current mismatches dI (these should stay the same)
# fundamental current difference only for nonlinear bus
dI_1 = Y_f.dot(V_f)[3] + G_bus4_1  # almost zero, different to Fuchs

# harmonic currents for all buses (including slack)
dI_5_nlin = Y_5.dot(V_5)[3] + G_bus4_5  # not zero, almost same as Fuchs
dI_5_lin = Y_5.dot(V_5)[:3]

# final dI
dI = np.array([dI_5_lin[0].real, dI_5_lin[0].imag,
               dI_5_lin[1].real, dI_5_lin[1].imag,
               dI_5_lin[2].real, dI_5_lin[2].imag,
               dI_5_nlin.real, dI_5_nlin.imag,
               dI_1.real, dI_1.imag])

# final harmonic mismatch vector
dM = np.append(dW, dI)
err_h = np.linalg.norm(dM, np.inf)

# start iteration
n_iter_h = 0
while err_h > 1e-6 and n_iter_h < 10:
    # J1 (dim = 4 x 6)
    J1 = J[:4, :]
    # J5 (dim = 4 x 8)
    # without nonlinear power balance now just zeros
    J5 = np.zeros((4, 8))

    # G51 (dim = 8 x 6)
    # derivatives of current injections (wrt fund), stays the same
    dgdt_1 = 0.9j*V.loc[(1, "bus4"), "V_m"]**3*\
             np.exp(3j*V.loc[(1, "bus4"), "V_p"])
    dgdV_1 = 0.9*V.loc[(1, "bus4"), "V_m"]**2*\
             np.exp(3j*V.loc[(1, "bus4"), "V_p"])
    G51 = np.zeros((8, 6))
    G51[6, 4] = dgdt_1.real
    G51[7, 4] = dgdt_1.imag
    G51[6, 5] = dgdV_1.real
    G51[7, 5] = dgdV_1.imag

    # Y55 + G55 (dim = 8 x 8)
    # derivatives of current injections (wrt h=5)
    dgdt_5 = 0.9j*V.loc[(5, "bus4"), "V_m"]**2 *\
             np.exp(3j*V.loc[(5, "bus4"), "V_p"])
    dgdV_5 = 0.6*V.loc[(5, "bus4"), "V_m"] *\
             np.exp(3j*V.loc[(5, "bus4"), "V_p"])
    Y55 = np.zeros((8, 8))
    G55 = np.zeros((8, 8))

    # fill in elements, see calculations on paper
    for i in range(0, 4):
        for k in range(0, 4):
            Y55[2*i, 2*k] = (1j*Y_5[i, k] * V_5[k]).real
            Y55[2*i+1, 2*k] = (1j*Y_5[i, k] * V_5[k]).imag
            Y55[2*i, 2*k+1] = (Y_5[i, k] *
                            np.exp(1j*V.loc[(5, "bus"+str(k+1)), "V_p"])).real
            Y55[2*i+1, 2*k+1] = (Y_5[i, k] *
                            np.exp(1j*V.loc[(5, "bus"+str(k+1)), "V_p"])).imag
            # could be outside, but for generalization later better here
            if i == 3 and k == 3:
                G55[2*i, 2*k] = dgdt_5.real
                G55[2*i+1, 2*k] = dgdt_5.imag
                G55[2*i, 2*k+1] = dgdV_5.real
                G55[2*i+1, 2*k+1] = dgdV_5.imag

    # Y11 + G11 (dim = 2 x 6)
    epsilon_1 = np.arctan(buses.at[(3, "Q_1")]/buses.at[(3, "P_1")])
    gamma_1 = V.at[(1, "bus4"), "V_p"] - epsilon_1  # current phase, bus 4

    Y11 = np.zeros((2, 6))
    for k in range(0, 3):
        Y11[0, 2*k] = (1j*Y_f[3, k+1]*V_f[k+1]).real
        Y11[1, 2*k] = (1j*Y_f[3, k+1]*V_f[k+1]).imag
        Y11[0, 2*k+1] = (Y_f[3, k+1] *
                         np.exp(1j*V.loc[(1, "bus"+str(k+2)), "V_p"])).real
        Y11[1, 2*k+1] = (Y_f[3, k+1] *
                         np.exp(1j*V.loc[(1, "bus"+str(k+2)), "V_p"])).imag
    # --> correct!

    G11 = np.zeros((2, 6))
    # Fuchs doesn't really build derivative, thus results differ
    dIdt_1 = -P_4_1/V.loc[(1, "bus4"), "V_m"]*np.exp(1j*gamma_1) * \
             2*np.sin(gamma_1 - V.loc[(1, "bus4"), "V_p"]) /\
             (np.cos(2*gamma_1 - 2*V.loc[(1, "bus4"), "V_p"]) + 1)
    dIdV_1 = -P_4_1/V.loc[(1, "bus4"), "V_m"]**2 * np.exp(1j*gamma_1) / \
             np.cos(V.loc[(1, "bus4"), "V_p"] - gamma_1)  # correct
    G11[0, 4] = dIdt_1.real
    G11[1, 4] = dIdt_1.imag
    G11[0, 5] = dIdV_1.real
    G11[1, 5] = dIdV_1.imag

    # G15 (dim = 2 x 8)
    G15 = np.zeros((2, 8))  # this was easy

    # assembling YG (dim = 10 x 14)
    YG = np.block([
        [G51, Y55+G55],
        [Y11+G11, G15]
    ])

    # H5 cut
    # H1 cut

    # assembling J_5 (dim 14 x 14)
    J_5 = np.block([
        [J1, J5],
        [G51, Y55+G55],
        [Y11+G11, G15]
    ])

    # 7.4.11: Computation of correction bus vector and iterating
    J_5_inv = np.linalg.inv(J_5)

    U_new = U - J_5_inv.dot(dM)  # "spsolve" better than inverting

    # update V
    V["V_p"][1:] = U_new[:14:2]
    V["V_m"][1:] = U_new[1:14:2]  # negative harmonic magnitudes
    V.loc[5, "V_p"] = np.array(V.loc[5, "V_p"]) + np.pi  # add pi to phase(p603)
    V.loc[5, "V_m"] = -np.array(V.loc[5, "V_m"])
    # TODO: this has influence on convergence properties, what to do?

    # create U by rearranging V
    UV = np.hstack(np.split(V[["V_p", "V_m"]], len(V)))
    # without control angles, cut off V_f of slack
    U = UV[0, 2:]

    # copy paste from here on, reduced comments

    # this whole part might not be needed anymore
    epsilon_1 = np.arctan(buses.at[(3, "Q_1")]/buses.at[(3, "P_1")])
    gamma_1 = V.at[(1, "bus4"), "V_p"] - epsilon_1  # current phase
    # fundamental "device currents" (why different to injection g(?))
    G_bus4_1_r = buses.at[(3, "P_1")]/pu_factor * np.cos(gamma_1) / \
                 (V.at[(1, "bus4"), "V_m"] *
                  np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1))
    G_bus4_1_i = buses.at[(3, "P_1")]/pu_factor * np.sin(gamma_1) / \
                 (V.at[(1, "bus4"), "V_m"] *
                  np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1))
    G_bus4_1 = G_bus4_1_r + 1j*G_bus4_1_i

    # now for h = 5
    G_bus4_5 = g(V, "bus4")  # difference between G and g?
    epsilon_5 = np.arctan(abs(G_bus4_5.imag)/abs(G_bus4_5.real))
    gamma_5 = V.at[(5, "bus4"), "V_p"] - epsilon_5
    G_bus4_5_r = abs(G_bus4_5)*np.cos(gamma_5)
    G_bus4_5_i = abs(G_bus4_5)*np.sin(gamma_5)
    # --> correct (except rounding and phase sign)

    # test
    P_4_1 = abs(G_bus4_1)*V.at[(1, "bus4"), "V_m"] * \
            np.cos(V.at[(1, "bus4"), "V_p"] - gamma_1)
    Q_4_1 = abs(G_bus4_1)*V.at[(1, "bus4"), "V_m"] * \
            np.sin(V.at[(1, "bus4"), "V_p"] - gamma_1)

    P_4_5 = abs(G_bus4_5)*V.at[(5, "bus4"), "V_m"] * \
            np.cos(V.at[(5, "bus4"), "V_p"] - gamma_5)
    Q_4_5 = abs(G_bus4_5)*V.at[(5, "bus4"), "V_m"] * \
            np.sin(V.at[(5, "bus4"), "V_p"] - gamma_5)

    # total injected powers at bus4
    P_4_t = P_4_1 + P_4_5
    Q_4_t = Q_4_1 + Q_4_5

    # 7.4.9: Evaluation of harmonic mismatch vector dM = [dW, dI_5, dI_1]
    # dW for the linear buses (analog to dm in 7.3.6):
    V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_p"])
    dW_lin = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)

    # dW for nonlinear buses needs harmonic line powers
    V_5 = V.loc[5, "V_m"]*np.exp(1j*V.loc[5, "V_p"])
    F_nlin = np.array(V_f*np.conj(Y_f.dot(V_f))) + \
             np.array(V_5*np.conj(Y_5.dot(V_5)))
    dW_nlin = F_nlin[3] + P_4_t + 1j*Q_4_t  # also different to Fuchs
    # final dW (excluding dW_nlin)
    dW = np.array([dW_lin[1].real, dW_lin[1].imag,
                   dW_lin[2].real, dW_lin[2].imag])

    # current mismatches dI (these should stay the same)
    # fundamental current difference only for nonlinear bus
    dI_1 = Y_f.dot(V_f)[3] + G_bus4_1  # almost zero, different to Fuchs

    # harmonic currents for all buses (including slack)
    dI_5_nlin = Y_5.dot(V_5)[3] + G_bus4_5  # not zero, almost same as Fuchs
    dI_5_lin = Y_5.dot(V_5)[:3]

    # final dI
    dI = np.array([dI_5_lin[0].real, dI_5_lin[0].imag,
                   dI_5_lin[1].real, dI_5_lin[1].imag,
                   dI_5_lin[2].real, dI_5_lin[2].imag,
                   dI_5_nlin.real, dI_5_nlin.imag,
                   dI_1.real, dI_1.imag])

    # final harmonic mismatch vector
    dM = np.append(dW, dI)
    err_h = np.linalg.norm(dM, np.inf)
    print("error_h: " + str(err_h))
    n_iter_h += 1

for i in V.loc[5].index:  # why did I do this? --> phase < 2pi
    V.loc[5, i] = A2P(P2A(V.loc[(5, i), "V_m"], V.loc[(5, i), "V_p"]))

print("ended after " + str(n_iter_h) + " iterations")
print("final voltages:")
print(V)

# surprised this actually worked out well
# converged after 10 iterations with very similar results, even smaller error
# but is this really correct? lost information because of reduced Jacobian?

