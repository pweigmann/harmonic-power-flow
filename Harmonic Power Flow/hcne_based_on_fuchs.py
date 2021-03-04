# Copy paste of 4 Bus example of "classical" harmonic load power based on Fuchs.
# Experiment to
# 1) cut away "control parameters" alpha and beta for nonlinear
# buses thus shortening dM, J_1, J_5 and U.
# 2) use Norton parameters to calculate the current injections

# Author: Pascal Weigmann, p.weigmann@posteo.de


import numpy as np
import pandas as pd

pu_factor = 1000
n_iter_max = 20
err_max = 0.0001
err_h_max = 0.01

# Norton parameter from fuchs:
Y_N = np.array([[9.78668336e-02-1.99148459e-01j,
                 -1.72493982e-05+1.76439133e-05j],
                [1.33136849e+01-5.84900472e+00j,
                 2.83733759e-02-1.23738247e-03j]])
I_N = np.array([0.11930918+0.11137297j, -12.94314986+5.8675826j])


# functions to change from algebraic to polar form
def P2A(radii, angles):
    return radii * np.exp(1j*angles)


def A2P(c):
    return abs(c), np.angle(c)


# initialize voltage as multi-index DataFrame, default: V = 1pu, V_h = 0.1pu
iterables = [[1, 5], ["bus1", "bus2", "bus3", "bus4"]]
multiIdx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
V = pd.DataFrame(np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
                 index=multiIdx, columns=["V_m", "V_a"])
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
                     columns=["ID", "fromID", "toID", "R", "X"])
S = buses["P"]/pu_factor+1j*buses["Q"]/pu_factor

# construct fundamental admittance matrix Y_f (7.3.5)
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
            # won't work if there are more than two lines per bus
            # simpler to use sum of values calculated above
            Y_f[n, m] = 1/(lines[lines.fromID == n+1].R.iloc[0] +
                           1j*lines[lines.fromID == n+1].X.iloc[0]) +\
                        (1/(lines[lines.toID == n+1].R.iloc[0] +
                            1j*lines[lines.toID == n+1].X.iloc[0]))


# 7.3.6: fundamental Newton Raphson algorithm: minimize mismatch vector f
# x_new = x - J^-1 * f
n_iter = 1
err = 1
while err > err_max and n_iter <= n_iter_max:
    # 7.3.6: fundamental mismatch vector dm (doesn't contain slack bus)
    V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])  # voltage as complex vec
    # power mismatch
    dm = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)

    # Newton Raphson variables (without slack bus)
    x = np.squeeze(np.hstack(np.split(
        V.loc[1][["V_a", "V_m"]], len(V.loc[1]))).transpose()[2:])
    f = np.zeros(2*(len(dm)-1))
    for n in range(1, len(dm)):
        f[2*n-2] = dm[n].real
        f[2*n-1] = dm[n].imag

    # 7.3.7: fundamental Jacobian (calculated the way pypsa does)
    I_diag = np.diag(Y_f.dot(V_f))
    V_diag = np.diag(V_f)
    V_diag_norm = np.diag(V_f/abs(V_f))

    # submatrices
    dSdt = 1j*V_diag.dot(np.conj(I_diag - Y_f.dot(V_diag)))
    dPdt = dSdt.real
    dQdt = dSdt.imag

    dSdV = V_diag_norm.dot(np.conj(I_diag)) + \
           V_diag.dot(np.conj(Y_f.dot(V_diag_norm)))
    dPdV = dSdV.real
    dQdV = dSdV.imag

    # sorting of jacobian entries differs from pypsa to fuchs, sorting:
    Jb = np.zeros((2*len(buses), 2*len(buses)))
    for n in range(len(buses)):
        for m in range(len(buses)):
            Jb[2*n, 2*m] = dPdt[n, m]
            Jb[2*n+1, 2*m] = dQdt[n, m]
            Jb[2*n, 2*m+1] = dPdV[n, m]
            Jb[2*n+1, 2*m+1] = dQdV[n, m]
    J = Jb[2:, 2:]  # without slack

    # 7.3.8: calculate the inverse of the Jacobian
    J_inv = np.linalg.inv(J)

    # 7.3.10: compute correction vector + iterate
    x_new = x - J_inv.dot(f)

    # update voltage:
    V.loc[1, 'V_a'][1:] = x_new[::2]
    V.loc[1, 'V_m'][1:] = x_new[1::2]

    err = np.linalg.norm(f, np.Inf)
    n_iter += 1
    print("error_f: " + str(err))

if err < err_max:
    print("Fundamental power flow converged after " +
          str(n_iter) + " iterations")
else:
    print("No convergence after " + str(n_iter_max) + " iterations!")


# HARMONIC POWER FLOW, HCNE version
# without nonlinear load control parameters alpha and beta
# --> unknowns reduced by 2 for each nonlinear bus, so also 2 less eq. needed
# --> remove total active and reactive power balances at nonlinear buses

# only 5th harmonic considered
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


# 7.4.8: Computation of nonlinear load harmonic currents
# in this case independent of device control parameters alpha and beta
def g(v, bus):
    g = 0.3*(v.at[(1, bus), "V_m"]**3)*np.exp(3j*v.at[(1, bus), "V_a"]) +\
        0.3*(v.at[(5, bus), "V_m"]**2)*np.exp(3j*v.at[(5, bus), "V_a"])
    return g


# extend "buses" to include harmonic parameters
buses.rename({"P": "P_1", "Q": "Q_1"}, axis=1, inplace=True)
buses = buses.assign(P_5=np.zeros(len(buses)), Q_5=np.zeros(len(buses)))

# start iteration
err_h = 1
n_iter_h = 0
V_h_log = {}
I_inj_log = {}
while err_h > err_h_max and n_iter_h < n_iter_max:
    V_h_log[n_iter_h] = V.copy()

    # create harmonic state vector U by rearranging V (p.280, eq. (7-93))
    UV = np.hstack(np.split(V[["V_a", "V_m"]], len(V)))
    # without control angles, cut off slack
    U = UV[0, 2:]
    # shorter notation
    V_f = V.loc[1, "V_m"]*np.exp(1j*V.loc[1, "V_a"])
    V_5 = V.loc[5, "V_m"]*np.exp(1j*V.loc[5, "V_a"])

    # nonlinear current injections G at bus4
    epsilon_1 = np.arctan(buses.at[(3, "Q_1")]/buses.at[(3, "P_1")])
    gamma_1 = V.at[(1, "bus4"), "V_a"] - epsilon_1
    # fundamental current injections
    # G referred to swing bus, g referred to bus4.
    # G and g are the same if gamma is small.
    G_bus4_1_r = buses.at[(3, "P_1")]/pu_factor * np.cos(gamma_1) / \
                 (V.at[(1, "bus4"), "V_m"] *
                  np.cos(V.at[(1, "bus4"), "V_a"] - gamma_1))
    G_bus4_1_i = buses.at[(3, "P_1")]/pu_factor * np.sin(gamma_1) / \
                 (V.at[(1, "bus4"), "V_m"] *
                  np.cos(V.at[(1, "bus4"), "V_a"] - gamma_1))
    G_bus4_1 = G_bus4_1_r + 1j*G_bus4_1_i

    # harmonic current injections
    g_bus4_5 = g(V, "bus4")
    epsilon_5 = np.arctan(abs(g_bus4_5.imag)/abs(g_bus4_5.real))
    gamma_5 = V.at[(5, "bus4"), "V_a"] - epsilon_5
    G_bus4_5_r = abs(g_bus4_5)*np.cos(gamma_5)
    G_bus4_5_i = abs(g_bus4_5)*np.sin(gamma_5)
    G_bus4_5 = g_bus4_5  # this is what yields same results as Fuchs

    # with this line wrong results, why? This is how Fuchs describes it (p.600)
    # G_bus4_5 = G_bus4_5_r + 1j*G_bus4_5_i

    # HCNE alternative: current injection from Norton Equivalents
    # V_bus4 = np.array([V_f[3], V_5[3]])
    # I_inj_NE = I_N - Y_N.dot(V_bus4)
    # G_bus4_1 = I_inj_NE[0]
    # G_bus4_5 = I_inj_NE[1]

    # 7.4.9: Evaluation of harmonic mismatch vector dM = [dW, dI_5, dI_1]
    # dW for the linear buses (analog to dm in 7.3.6):
    dW_lin = np.array(V_f*np.conj(Y_f.dot(V_f))) + np.array(S)

    # final dW (now without dW_nlin)
    dW = np.array([dW_lin[1].real, dW_lin[1].imag,
                   dW_lin[2].real, dW_lin[2].imag])

    # current mismatches dI
    # fundamental current difference only for nonlinear bus
    dI_1 = Y_f.dot(V_f)[3] + G_bus4_1

    # harmonic currents for all buses (including slack)
    dI_5_nlin = Y_5.dot(V_5)[3] + G_bus4_5
    dI_5_lin = Y_5.dot(V_5)[:3]

    # log current injections
    I_inj_log[n_iter_h] = pd.DataFrame([[G_bus4_1.real, G_bus4_1.imag],
                                        [G_bus4_5.real, G_bus4_5.imag]],
                                       index=[1, 5])

    # final dI
    dI = np.array([dI_5_lin[0].real, dI_5_lin[0].imag,
                   dI_5_lin[1].real, dI_5_lin[1].imag,
                   dI_5_lin[2].real, dI_5_lin[2].imag,
                   dI_5_nlin.real, dI_5_nlin.imag,
                   dI_1.real, dI_1.imag])

    # final harmonic mismatch vector
    dM = np.append(dW, dI)
    err_h = np.linalg.norm(dM, np.inf)

    # extended Jacobian
    # J1 (dim = 4 x 6)
    J1 = J[:4, :]
    # J5 (dim = 4 x 8)
    # without nonlinear power balance now just zero
    J5 = np.zeros((4, 8))

    # G51 (dim = 8 x 6)
    # derivatives of current injections (wrt fund)
    dgdt_1 = 0.9j*V.loc[(1, "bus4"), "V_m"]**3*\
             np.exp(3j*V.loc[(1, "bus4"), "V_a"])
    dgdV_1 = 0.9*V.loc[(1, "bus4"), "V_m"]**2*\
             np.exp(3j*V.loc[(1, "bus4"), "V_a"])
    G51 = np.zeros((8, 6))
    G51[6, 4] = dgdt_1.real
    G51[7, 4] = dgdt_1.imag
    G51[6, 5] = dgdV_1.real
    G51[7, 5] = dgdV_1.imag

    # Y55 + G55 (dim = 8 x 8)
    # derivatives of current injections (wrt h=5)
    dgdt_5 = 0.9j*V.loc[(5, "bus4"), "V_m"]**2 *\
             np.exp(3j*V.loc[(5, "bus4"), "V_a"])
    dgdV_5 = 0.6*V.loc[(5, "bus4"), "V_m"] *\
             np.exp(3j*V.loc[(5, "bus4"), "V_a"])
    Y55 = np.zeros((8, 8))
    G55 = np.zeros((8, 8))

    # fill in elements, see calculations on paper
    for i in range(0, 4):
        for k in range(0, 4):
            Y55[2*i, 2*k] = (1j*Y_5[i, k] * V_5[k]).real
            Y55[2*i+1, 2*k] = (1j*Y_5[i, k] * V_5[k]).imag
            Y55[2*i, 2*k+1] = (Y_5[i, k] *
                            np.exp(1j*V.loc[(5, "bus"+str(k+1)), "V_a"])).real
            Y55[2*i+1, 2*k+1] = (Y_5[i, k] *
                            np.exp(1j*V.loc[(5, "bus"+str(k+1)), "V_a"])).imag
            # could be outside, but for generalization later better here
            if i == 3 and k == 3:
                G55[2*i, 2*k] = dgdt_5.real
                G55[2*i+1, 2*k] = dgdt_5.imag
                G55[2*i, 2*k+1] = dgdV_5.real
                G55[2*i+1, 2*k+1] = dgdV_5.imag

    # Y11 + G11 (dim = 2 x 6)
    epsilon_1 = np.arctan(buses.at[(3, "Q_1")]/buses.at[(3, "P_1")])
    gamma_1 = V.at[(1, "bus4"), "V_a"] - epsilon_1  # current phase, bus 4

    Y11 = np.zeros((2, 6))
    for k in range(0, 3):
        Y11[0, 2*k] = (1j*Y_f[3, k+1]*V_f[k+1]).real
        Y11[1, 2*k] = (1j*Y_f[3, k+1]*V_f[k+1]).imag
        Y11[0, 2*k+1] = (Y_f[3, k+1] *
                         np.exp(1j*V.loc[(1, "bus"+str(k+2)), "V_a"])).real
        Y11[1, 2*k+1] = (Y_f[3, k+1] *
                         np.exp(1j*V.loc[(1, "bus"+str(k+2)), "V_a"])).imag

    G11 = np.zeros((2, 6))
    # corrected and simplified version of dIdt_1
    dIdt_1 = 1j*G_bus4_1
    dIdV_1 = -G_bus4_1/V.loc[(1, "bus4"), "V_m"]

    G11[0, 4] = dIdt_1.real
    G11[1, 4] = dIdt_1.imag
    G11[0, 5] = dIdV_1.real
    G11[1, 5] = dIdV_1.imag

    # G15 (dim = 2 x 8)
    G15 = np.zeros((2, 8))

    # assembling YG (dim = 10 x 14)
    YG = np.block([
        [G51, Y55+G55],
        [Y11+G11, G15]
    ])

    # H5 cut
    # H1 cut

    # assembling J_5 (dim 14 x 14)
    J_5 = np.block([[J1, J5],
                    [G51, Y55+G55],
                    [Y11+G11, G15]])

    # 7.4.11: Newton-Raphson step
    # Computation of correction bus vector and iterating
    J_5_inv = np.linalg.inv(J_5)
    U_new = U - J_5_inv.dot(dM)  # "solve" function better than inverting

    # update V
    V["V_a"][1:] = U_new[:14:2]
    V["V_m"][1:] = U_new[1:14:2]  # negative harmonic magnitudes (avoidable?)
    V.loc[5, "V_a"] = np.array(V.loc[5, "V_a"]) + np.pi  # add pi to phase(p603)
    V.loc[5, "V_m"] = -(np.array(V.loc[5, "V_m"]))  # ...and change signum

    # end iteration
    print("error_h: " + str(err_h))
    n_iter_h += 1

for i in V.loc[5].index:  # ensure phase < 2pi
    V.loc[5, i] = A2P(P2A(V.loc[(5, i), "V_m"], V.loc[(5, i), "V_a"]))

if err_h < err_h_max:
    print("Harmonic power flow converged after " +
          str(n_iter_h) + " iterations")
else:
    print("No convergence after " + str(n_iter_max) + " iterations")

print("final voltages:")
print(V)

pd.concat(V_h_log, names=["iteration"]).to_json("V_log.json", orient="table")
pd.concat(I_inj_log, names=["iteration", "harmonic"]).to_json(
    "I_log.json", orient="table")

# Results without alpha and beta are the same to original example.
# Convergence seems to be even better (not faster, but further).
# Problem might be the constant fundamental Jacobian
