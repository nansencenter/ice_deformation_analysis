#!/usr/bin/env python
import sys

import numpy as np

from pysida.lib import MeshFileList, merge_pairs

# nominal neXtSIM mesh resolution
resolution = 10000
# minimal a/p ratio
r_min = 0.13
# maximal area
a_max = 2. * resolution ** 2
# distance from RGPS nodes to neXtSIM nodes for initial subset
distance_upper_bound1 = 100000
# distance from RGPS elements to neXtSIM elements for final subset
distance_upper_bound2 = resolution * 1.5
cores = 10

# READ script inputs
rfile = str(sys.argv[1])
idir = str(sys.argv[2])
ofile = str(sys.argv[3])

r_pairs = np.load(rfile, allow_pickle=True)['pairs']

mfl = MeshFileList(idir, lazy=True)
n_pairs = merge_pairs(r_pairs, mfl, r_min, a_max, distance_upper_bound1, distance_upper_bound2, cores=cores)
np.savez(ofile, pairs=n_pairs)
