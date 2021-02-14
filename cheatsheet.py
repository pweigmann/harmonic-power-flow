# cheatsheet for remembering how to use multi-index DataFrames

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Multi-index handling
# example multi-index DataFrame
iterables = [[1, 5], ["bus1", "bus2", "bus3", "bus4"]]
multiIdx = pd.MultiIndex.from_product(iterables, names=['harmonic', 'bus'])
V = pd.DataFrame(np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
                 index=multiIdx, columns=["V_m", "V_p"])
# V.sort_index(inplace=True)

# analyse df
V.index
V.columns
V.head()  # show first 5 rows

# manipulate index
# V.reset_index()
# V.set_index()
# V.sort_index()  # necessary for slicing
# V.swaplevel()  # swap indices
# V.unstack()  # expand one index level into new column level


# direct indexing
# return column as series
V["V_m"]
# ... or a list of columns as df
V[["V_m", "V_p"]]
# alternative: slicing along the other axis with .loc
V.loc(axis=1)["V_m"]

# .loc
# value of first level index (h = 1), index cut of
V.loc[1]
# value of first level index (h = 1), index NOT cut of
V.loc[[1]]

# (interpreted as) tuple for both indices, returns series, index cut of
V.loc[1, "bus1"]
# (must be) tuple(s) for both indices, returns list of rows as DataFrame
V.loc[[(1, "bus1")]]

# tuple for both indices, name of column, returns single value
V.loc[(1, "bus1"), "V_m"]
# ... or list of values as series
V.loc[[(1, "bus1"), (1, "bus2")], "V_m"]




# .iloc
# first row of df, returns series (returns df with .iloc[[0]])
V.iloc[0]

# value of first row, first column
V.iloc[0, 0]

# .at
# like .loc but for getting and setting single values (.iat analog)
V.at[(1, "bus1"), "V_m"]
V.iat[2, 0]

# .xs
# cross section, which takes "level" argument (no writing)
# e.g. for selecting by second index of multi-index
V.xs("bus1", level=1, drop_level=False)

# same, but using slices
V.loc[(slice(None), "bus1"), :]
V.loc[(slice(None), "bus1"), "V_m"]

# most "matlab" like + flexible, typically abbreviated by "idx = pd.IndexSlice"
V.loc[pd.IndexSlice[:, 'bus1'], :]

# .groupby
# calculate mean of each harmonic (find number with .size() )
V.groupby(["harmonic"]).mean()


# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 0], [0, 8]])
Csp = csr_matrix([[5, 0], [0, 8]])  # sparse matrix

# element wise multiplication
A * B

# matrix-multiplication
A.dot(B)

# sparse multiplication is matrix-multiplication!
A * Csp

# this returns matrix of sparse matrices!
A.dot(Csp)

