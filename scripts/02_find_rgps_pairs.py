#!/usr/bin/env python
from datetime import datetime
import sys

import pandas as pd
import numpy as np

from pysida.lib import get_rgps_pairs

ifile = str(sys.argv[1])
ofile = ifile.replace('_LP.df', '_pairs.npz')
print(ofile)

date0 = datetime.strptime(str(sys.argv[2]), '%Y-%m-%d')
date1 = datetime.strptime(str(sys.argv[3]), '%Y-%m-%d')

min_time_diff = 0.5
max_time_diff = 7
min_size = 100
r_min = 0.12
a_max = 200e6
cores = 5

df = pd.read_pickle(ifile)
pairs = get_rgps_pairs(df, date0, date1, min_time_diff=min_time_diff, max_time_diff=max_time_diff, min_size=min_size, r_min=r_min, a_max=a_max, cores=cores)
print('N pairs', len(pairs))
np.savez(ofile, pairs=pairs)