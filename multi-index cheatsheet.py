# cheatsheet for remembering how to use multi-index DataFrames

import numpy as np
import pandas as pd

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
V.reset_index()
V.set_index()
V.sort_index()  # necessary for slicing
V.swaplevel()  # swap indices
V.unstack()  # expand one index level into new column level


# direct indexing
# return column as series
V["V_m"]
# ... or a list of columns as df
V[["V_m", "V_p"]]

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

# returns a list of rows as df
V.iloc


# groupby
# calculate mean of each harmonic (find number with .size() )
V.groupby(["harmonic"]).mean()

