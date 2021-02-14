# Importing Simulink simulation results from .mat file and calculating the
# corresponding coupled and uncoupled Norton equivalents.

import numpy as np
import pandas as pd
from scipy.io import loadmat

# import from .mat file
data = loadmat('circuit_sim.mat', squeeze_me=True, struct_as_record=False)

# convert to multi-index DataFrame
supply_frequencies = []
for n in data["results"]:
    supply_frequencies.append(n[0].f_h)

supply_voltages = []
for m in data["results"][0, :]:
    supply_voltages.append(m.V_m_h)

iterables = [supply_voltages, supply_frequencies]
multi_idx = pd.MultiIndex.from_product(iterables,
                                       names=['supply_v_h', 'supply_f_h'])
spectrum = pd.Int64Index(data["results"][0, 0].H.astype(int),
                         name="Frequency", dtype="int64")

# initialize multi-index DataFrame
I_inj = pd.DataFrame(np.zeros((len(supply_frequencies)*len(supply_voltages),
                               len(data["results"][0, 0].H))),
                     index=multi_idx, columns=spectrum)

# iterate through measurements
for i in data["results"]:
    for j in i:
        # write I_inj of corresponding voltage to df
        I_inj.loc[j.V_m_h, j.f_h] = j.I_inj

# final df, only at harmonic frequencies
I_inj_h = I_inj.loc(axis=1)[50::4]
