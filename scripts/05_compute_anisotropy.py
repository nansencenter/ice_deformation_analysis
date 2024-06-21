#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from pysida.lib import defor_to_aniso_con

# minimum total deformation to compute anisotropy from
min_e = 0.05
# at which distances anisotropy should be computed
edges_vec = [2,3,4,5,6,7,8,9]
# which power to use in the inertia matrix computation
power = 0.5
# minimum size of a pair to compute anisotropy
min_size = 100
# parallelizm
cores = 5

# read script input
pfile = str(sys.argv[1])

dfile = pfile.replace('pairs.npz', 'defor.npz')
afile = pfile.replace('pairs.npz', 'aniso.npz')
pairs = np.load(pfile, allow_pickle=True)['pairs']
defor = np.load(dfile, allow_pickle=True)['defor']

defor_to_aniso_con.min_e = min_e
defor_to_aniso_con.edges_vec = edges_vec
defor_to_aniso_con.power = power
defor_to_aniso_con.min_size = min_size
with Pool(cores) as p:
    aniso = p.map(defor_to_aniso_con, zip(pairs, defor))

np.savez(afile, aniso=aniso)