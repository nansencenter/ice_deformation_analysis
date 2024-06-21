#!/usr/bin/env python
from multiprocessing import Pool
import sys

import numpy as np
from skimage.feature import graycoprops

from pysida.lib import get_glcmd

cores = 5

# number of gray levels
l = 10
# minimum and maximum deformation (log scale)
e_min = -4
e_max = -0.7
# distances for GLCM computation
d_vec = [1,2,4,8]
# names of texture features
propnames = [
    'dissimilarity',
    'homogeneity',
    'ASM',
    'energy',
    'correlation',
    'contrast',
]

pfile = str(sys.argv[1])

dfile = pfile.replace('pairs.npz', 'defor.npz')
ofile = pfile.replace('pairs.npz', 'texture.npz')

with np.load(pfile, allow_pickle=True) as f:
    pairs = f['pairs']
with np.load(dfile, allow_pickle=True) as f:
    defor = f['defor']

get_glcmd.l = l
get_glcmd.e_min = e_min
get_glcmd.e_max = e_max
get_glcmd.d_vec = d_vec
with Pool(cores) as p:
    glcmds = p.map(get_glcmd, zip(pairs, defor))

p = []
for glcmd in glcmds:
    if glcmd is None:
        p.append(np.zeros((6,4,1))+np.nan)
    else:
        p.append(np.array([graycoprops(glcmd, propname) for propname in propnames]))
props = np.array(p)[:,:,:,0]
np.savez(ofile, props=props)