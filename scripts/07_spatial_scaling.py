#!/usr/bin/env python
from multiprocessing import Pool
import os
import sys

import numpy as np
import pandas as pd

from pysida.lib import PairFilter, MarsanSpatialScaling

# path to pairs file
pfile = str(sys.argv[1])
date_begin = pd.Timestamp(sys.argv[2]) #2006-12-05
date_end = pd.Timestamp(sys.argv[3]) # 2007-05-15
# frequency at which to compute deformation snapshots, e.g. '1D'
freq = str(sys.argv[4])
# nominal mesh resolution
resolution = int(sys.argv[5]) # 10000
# parallelizm
cores = 5

dist2coast_path = '/Data/sim/data/BATHYMETRY/dist2coast_4deg.npy'

dfile = pfile.replace('pairs.npz', 'defor.npz')
ofile = pfile.replace('pairs.npz', 'scale.npz')
pairs = np.load(pfile, allow_pickle=True)['pairs']
defor = np.load(dfile, allow_pickle=True)['defor']

pf = PairFilter(pairs, defor, resolution, dist2coast_path=dist2coast_path)
mss = MarsanSpatialScaling(pf, resolution)

dates = pd.date_range(date_begin, date_end, freq=freq).to_pydatetime()
with Pool(cores) as p:
    mmm = p.map(mss.proc_one_date, dates)

np.savez(ofile, dates=dates, mmm=mmm)