# rewriting the harmonic coupled norton equivalent method in a generalized and
# modularized way

import numpy as np
import pandas as pd

# global variables

pu_factor = 1000
harmonics = np.array([5, 7, 11])


# infrastructure (TODO: import from file)
buses = pd.DataFrame(np.array([[1, "slack", "generator", 0, 0, 1000, 0.0001],
                               [2, "PQ", "lin_load_1", 100, 100, None, 0],
                               [3, "PQ", None, 0, 0, None, 0],
                               [4, "nonlinear", "nlin_load_1",
                                250, 100, None, 0],
                               [5, "nonlinear", "nlin_load_1",
                                250, 100, None, 0]]),
                     columns=["ID", "type", "component", "P1",
                              "Q1", "S1", "X_shunt"])
lines = pd.DataFrame(np.array([[1, 1, 2, 0.01, 0.01],
                               [2, 2, 3, 0.02, 0.08],
                               [3, 3, 4, 0.01, 0.02],
                               [4, 4, 5, 0.01, 0.02],
                               [5, 5, 1, 0.01, 0.02]]),
                     columns=["ID", "fromID", "toID", "R", "X"])




