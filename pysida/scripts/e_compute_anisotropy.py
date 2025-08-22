#!/usr/bin/env python
import sys

import numpy as np
from multiprocessing import Pool

from pysida.lib import DeformationToAnisotropyConnected, BaseRunner

class Runner(BaseRunner):
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
    # minimum number of neighbors to compute anisotropy
    min_neibors = 3

    def __call__(self, pfile):
        dfile = pfile.replace('pairs.npz', 'defor.npz')
        ofile = pfile.replace('pairs.npz', 'aniso.npz')
        if self.skip_processing(ofile): return ofile
        pairs = np.load(pfile, allow_pickle=True)['pairs']
        defor = np.load(dfile, allow_pickle=True)['defor']
        defor_to_aniso_con = DeformationToAnisotropyConnected(
            min_e=self.min_e, edges_vec=self.edges_vec, power=self.power, min_size=self.min_size, min_neibors=self.min_neibors)
        with Pool(self.cores) as p:
            aniso = p.map(defor_to_aniso_con, zip(pairs, defor))
        print(ofile)
        np.savez(ofile, aniso=aniso)


if __name__ == '__main__':
    # read script input
    pfile = str(sys.argv[1])
    Runner()(pfile)
