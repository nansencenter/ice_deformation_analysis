#!/usr/bin/env python
import sys

import numpy as np

from pysida.lib import MeshFileList, merge_pairs, BaseRunner

class Runner(BaseRunner):
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
    cores = 5

    def __call__(self, rfile, idir, ofile):
        if self.skip_processing(ofile): return ofile
        r_pairs = np.load(rfile, allow_pickle=True)['pairs']
        mfl = MeshFileList(idir, lazy=True)
        n_pairs = merge_pairs(
            r_pairs,
            mfl,
            self.r_min,
            self.a_max,
            self.distance_upper_bound1,
            self.distance_upper_bound2,
            cores=self.cores
        )
        print(ofile, 'N pairs:', len(n_pairs))
        np.savez(ofile, pairs=n_pairs)
        return ofile

if __name__ == '__main__':
    rfile = str(sys.argv[1])
    idir = str(sys.argv[2])
    ofile = str(sys.argv[3])
    Runner()(rfile, idir, ofile)