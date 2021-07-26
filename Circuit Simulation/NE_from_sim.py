# Importing SimuLink simulation results from .mat file and calculating the
# corresponding coupled and uncoupled Norton equivalents.

import numpy as np
import pandas as pd
from scipy.io import loadmat

# TODO
#  - change angles consistently to degree or rad or go full cartesian
#  - create and use V_supply in same shape and with same index as I_inj
#  - join all data initially by using loop, so it can be accessed vectorized
#  - restructure as functions, call for all files in folder
#  - pass name of simulated device (other meta data as well?)
#  - avoid hardcoded information (like base frequency)

# import from .mat file
device_name = "SMPS"
f_max = 5050

#def get_NE_from_sim(device_name, f_max):
''' Data structure - harmonic (dh) and fundamental (df)
dh[a, b]: a is index for harmonic, varying harmonic voltage source frequency
          b is index for measurement, varying harmonic voltage source magnitude
          Minimum values for uncoupled script to work: a > 0, b > 0
          Example: dh[0, 1] means first harmonic frequency (f_h = 150),
                   second measurement (e.g. V_m_h = 2.3).
df[c]:    c is index for measurement, varying fundamental voltage source angle
'''
data = loadmat(device_name + "_" + str(f_max) + ".mat",
               squeeze_me=True, struct_as_record=False)
df = data["all"].results_f  # fundamental simulation results
dh = data["all"].results_h  # harmonic simulation results


# --- Data Cleaning ---
# check imported data size
if len(dh[0, :]) < 2:
    raise ValueError('At least 2 measurements needed for script to work.')
elif len(dh[:, 0]) < 2:
    raise ValueError('At least 2 harmonics needed for script to work.')

# convert to multi-index DataFrame
supply_f = []
for n in dh:
    supply_f.append(n[0].f_h)

supply_v_m = []
for m in dh[0, :]:
    supply_v_m.append(m.V_m_h)

# this will only work for constant harmonic supply voltage angles
supply_v_a = [dh[0, 0].V_a_h]

iterables = [supply_v_m, supply_f, supply_v_a]
multi_idx = pd.MultiIndex.from_product(iterables, names=['supply_v_m',
                                                         'supply_f',
                                                         'supply_v_a'])
spectrum = pd.Int64Index(dh[0, 0].H.astype(int), name="Frequency",
                         dtype="int64")

# initialize multi-index DataFrame
I_inj_complete = pd.DataFrame(np.zeros((len(supply_f)*len(supply_v_m),
                                        len(dh[0, 0].H))), index=multi_idx,
                              columns=spectrum)
V_supply = pd.DataFrame(np.zeros((len(supply_f)*len(supply_v_m), 1)),
                        index=multi_idx)
# iterate through harmonic measurements
for i in dh:
    for j in i:
        # write I_inj of corresponding voltage to DataFrame
        I_inj_complete.loc[(j.V_m_h, j.f_h, j.V_a_h)] = \
            j.I_inj*np.exp(1j*j.I_inj_phase)
        V_supply.loc[(j.V_m_h, j.f_h, j.V_a_h)] = \
            j.V_m_h*np.exp(1j*j.V_a_h*np.pi/180)

# iterate through fundamental measurements
for s in df:
    # write I_inj of corresponding voltage to DataFrame, f_fund always 50 Hz
    I_inj_complete.loc[(s.V_m_f, 50, s.V_a_f)] = \
        s.I_inj*np.exp(1j*s.I_inj_phase)

# final I_inj for all measurements, no inter-harmonics
I_inj = I_inj_complete.loc(axis=1)[50::dh[0, 0].cycles*2]


# --- calculate Norton Equivalent parameters, UNCOUPLED ---
# (see Thunberg.1999), 2*n measurements (of 1 freq.) for n harmonics
# build difference between the two measurements, without fund, m2 - m1
dI_h = (I_inj.loc[dh[0, 1].V_m_h] - I_inj.loc[dh[0, 0].V_m_h]).drop(
    50, axis=1)
# voltage difference is constant wrt frequency
V_h_m1 = dh[0, 0].V_m_h*np.exp(1j*dh[0, 0].V_a_h*np.pi/180)
V_h_m2 = dh[0, 1].V_m_h*np.exp(1j*dh[0, 1].V_a_h*np.pi/180)
# only diagonal elements needed for harmonic frequencies
# Norton admittance, harmonic
Y_N_h = pd.Series(np.diag(dI_h)/(V_h_m1 - V_h_m2), index=supply_f)
Y_N_h.rename_axis(index="Frequency", inplace=True)
# Norton current source, harmonic:
I_N_h = Y_N_h*V_h_m1 + np.diag(I_inj.loc[dh[0, 0].V_m_h, 150:])

# build difference between the two measurements, only fund, m2 - m1
dI_f = I_inj.loc[(df[1].V_m_f, 50, df[1].V_a_f)] - \
       I_inj.loc[(df[0].V_m_f, 50, df[0].V_a_f)]
# voltage difference
V_f_m1 = df[0].V_m_f*np.exp(1j*df[0].V_a_f*np.pi/180)
V_f_m2 = df[1].V_m_f*np.exp(1j*df[1].V_a_f*np.pi/180)
# only fund injection needed for fundamental frequency
Y_N_f = dI_f[[50]]/(V_f_m1 - V_f_m2)  # Norton admittance, fundamental
# Norton current source, fundamental:
I_N_f = Y_N_f*V_f_m1 + I_inj.loc[(df[0].V_m_f, 50, df[0].V_a_f), [50]]

# Final Norton parameters, uncoupled model:
I_N_uc = I_N_f.append(I_N_h)
Y_N_uc = Y_N_f.append(Y_N_h)

# test uncoupled (using measurement 1, dU = 0.005pu )
# calculate I_inj with NE
V1 = np.array([V_f_m1] + len(supply_f)*[V_h_m1])
I_inj_test = I_N_uc - np.diag(Y_N_uc) @ V1
# take I_inj from circuit sim
I_inj_m1 = pd.Series(I_inj.loc[(df[0].V_m_f, 50, df[0].V_a_f), [50]]).append(
    pd.Series(np.diagonal(I_inj.loc[dh[0, 0].V_m_h], 1), index=supply_f))
err = I_inj_test - I_inj_m1

# test (using measurement 2, dU = 0.01pu)
V2 = np.array([V_f_m2] + len(supply_f)*[V_h_m2])
I_inj_test_2 = I_N_uc - np.diag(Y_N_uc) @ V2
I_inj_m2 = pd.Series(I_inj.loc[(df[1].V_m_f, 50, df[1].V_a_f), [50]]).append(
    pd.Series(np.diagonal(I_inj.loc[dh[0, 1].V_m_h], 1), index=supply_f))
err2 = I_inj_test_2 - I_inj_m2

if np.linalg.norm((err, err2), np.inf) > 1e-6:
    print("Warning: uncoupled NE test failed!")
else:
    print("Uncoupled parameters passed consistency test.")


# --- calculate Norton Equivalent parameters, COUPLED ---
# coupled (see Almeida.2010), n+1 measurements (of all freq.) for n harmonics
# only one value for V_m_h needed (aka. one "measurement" of uncoupled NE)
freq = [50] + supply_f  # all frequencies used as supply voltages
N = len(freq)  # number of frequencies of which NE can be calculated

# constant V matrix for calculating coupled NEs
V_mes = pd.DataFrame(np.zeros((N+1, N)), index=freq + [50], columns=freq)
V_mes["I"] = -np.ones((N+1, 1))
V_mes[50] = V_f_m1  # fundamental voltage constant throughout measurements...
# ...except for N+1th. Second measurement at fundamental frequency placed at end
V_mes.iloc[-1, 0] = V_f_m2
# fill in harmonic supply voltages diagonally
for h in supply_f:
    V_mes.loc[h, h] = V_supply.loc[(supply_v_m[0], h, supply_v_a[0]), 0]
V_mes = -V_mes
V_inv = np.linalg.inv(V_mes)

# DataFrame to store results
YI_N = pd.DataFrame(np.zeros((N+1, N)), index=freq + ["I"], columns=freq)

# sort and slice I_inj, so the columns can be used directly
# first fund. I_inj
d1 = I_inj.loc[[(df[0].V_m_f, 50, df[0].V_a_f)], freq]
# harmonic I_inj
d2 = I_inj.loc[(dh[0, 0].V_m_h, slice(None), dh[0, 0].V_a_h), freq]
# second fund. I_inj
d3 = I_inj.loc[[(df[1].V_m_f, 50, df[1].V_a_f)], freq]
# concatenate DataFrames in correct order
I_inj_c = d1.append([d2, d3])
# evaluate NE, Y_N are added transposed
YI_N[freq] = V_inv.dot(I_inj_c[freq])

# final coupled Norton Equivalents, transpose Y_N back
Y_N_c = np.transpose(YI_N.iloc[:-1])  # (Y_N_ff equal to uncoupled)
I_N_c = YI_N.iloc[-1]  # (fundamental current source equal to uncoupled)

# spectrum as used for OpenDSS
spectrum_m = abs(I_inj.iloc[-2, :])/abs(I_inj.iloc[-2, 0])
spectrum_a = np.angle(I_inj.iloc[-2, :])*180/np.pi
spec = pd.DataFrame({"Harmonic": [i/50 for i in freq],
                     "I_m": spectrum_m,
                     "I_a": spectrum_a})

# test coupled NE for h = 5
V_uc_test5 = np.zeros(len(freq), dtype=complex)
V_uc_test5[0] = V_f_m1
V_uc_test5[2] = V_h_m1
I_inj_c_test5 = I_N_c - Y_N_c.dot(V_uc_test5)  # calculate I_inj with NE
err_c = I_inj_c_test5 - I_inj.loc[(dh[1, 0].V_m_h, dh[1, 0].f_h,
                                   dh[1, 0].V_a_h)]

if np.linalg.norm(err_c, np.inf) > 1e-6:
    print("Warning: coupled NE test failed!")
else:
    print("Coupled parameters passed consistency test.")

# prepare to export to file
multi_idx_exp = pd.MultiIndex.from_arrays([
    len(freq)*["Y_N_c"] + ["I_N_c", "Y_N_uc", "I_N_uc"],
    freq + [0, 0, 0]], names=["Parameter", "Frequency"])
NE = pd.DataFrame(np.zeros((len(freq)+3, len(freq))),
                  index=multi_idx_exp, dtype=complex, columns=freq)
NE.loc["Y_N_c"] = Y_N_c.to_numpy()
NE.loc["I_N_c", 0] = I_N_c.values
NE.loc["Y_N_uc", 0] = Y_N_uc.values
NE.loc["I_N_uc", 0] = I_N_uc.values

# export NE as needed for HarmonicPowerFlow.jl
filename = device_name + "_" + str(f_max) + "_NE.csv"
NE.to_csv(filename)
print("Created file " + filename)

# export opendss spectrum
filename_opendss = "spectrum_" + device_name + "_" + str(f_max) + ".csv"
spec.to_csv(filename_opendss, header=None, index=None)
print("Created file " + filename_opendss)
