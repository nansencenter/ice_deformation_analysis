#!/usr/bin/env python
from multiprocessing import Pool
import sys


import numpy as np

from pysida.lib import get_deformation_from_pair

cores = 5

pfile = str(sys.argv[1])
ofile = pfile.replace('_pairs.npz', '_defor.npz')

pairs = np.load(pfile, allow_pickle=True, fix_imports=False)['pairs']
with Pool(cores) as p:
    defor = p.map(get_deformation_from_pair, pairs)

np.savez(ofile, defor=np.array(defor, dtype='object'))