# Importing Simulink simulation results from .mat file and calculating the
# corresponding coupled and uncoupled Norton equivalents.

import numpy as np
import pandas as pd
from scipy.io import loadmat

# import from .mat file
data = loadmat('circuit_sim.mat', squeeze_me=True, struct_as_record=False)

supply_frequencies = []
for n in data["results"]:
    supply_frequencies.append(n[0].f_h)

# convert to multi-index DataFrame
supply_voltages = [data["results"][0, 0].V_m_h, data["results"][0, 1].V_m_h]
iterables = [supply_voltages, supply_frequencies]
multi_idx = pd.MultiIndex.from_product(iterables,
                                       names=['supply_v_h', 'supply_f_h'])
spectrum = pd.Int64Index(data["results"][0, 0].H.astype(int),
                         name="Frequency", dtype="int64")

# initialize multi-index DataFrame
I_inj = pd.DataFrame(np.zeros((len(supply_frequencies)*2,
                               len(data["results"][0, 0].H))),
                     index=multi_idx, columns=spectrum)

# iterate through measurements
for n in data["results"]:
    for m in n:
        # write I_inj of corresponding voltage to df
        I_inj.loc[m.V_m_h, m.f_h] = m.I_inj
