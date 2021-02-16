# Importing SimuLink simulation results from .mat file and calculating the
# corresponding coupled and uncoupled Norton equivalents.

# needs exactly 2 levels of variable simulation parameters
#  default: harmonic supply voltage magnitude V_m_h and frequency f_h

import numpy as np
import pandas as pd
from scipy.io import loadmat

# import from .mat file
data = loadmat('circuit_sim.mat', squeeze_me=True, struct_as_record=False)
df = data["all"].results_f  # fundamental simulation results
dh = data["all"].results_h  # harmonic simulation results

# convert to multi-index DataFrame
# TODO: enable single measurement support or raise error if only one measurement
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

# iterate through harmonic measurements
for i in dh:
    for j in i:
        # write I_inj of corresponding voltage to DataFrame
        I_inj_complete.loc[(j.V_m_h, j.f_h, j.V_a_h)] = \
            j.I_inj*np.exp(1j*j.I_inj_phase)

# iterate through fundamental measurements
for s in df:
    # write I_inj of corresponding voltage to DataFrame, f_fund always 50 Hz
    I_inj_complete.loc[(s.V_m_f, 50, s.V_a_f)] = \
        s.I_inj*np.exp(1j*s.I_inj_phase)

# final I_inj for all measurements, no inter-harmonics
I_inj = I_inj_complete.loc(axis=1)[50::dh[0, 0].cycles*2]


# calculate Norton Equivalent parameters
# uncoupled (see Thunberg.1999), 2*n measurements (of 1 freq.) for n harmonics
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
I_N_h = Y_N_h*V_h_m1 + I_inj.loc[(dh[0, 0].V_m_h, 150), 150:]


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
I_N_uc = I_N_f.append(I_N_h.loc[30])
Y_N_uc = Y_N_f.append(Y_N_h)


# coupled (see Almeida.2010), n+1 measurements (of all freq.) for n harmonics

