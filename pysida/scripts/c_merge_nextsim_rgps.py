#!/usr/bin/env python
import sys

import numpy as np

from pysida.lib import MeshFileList, merge_pairs, BaseRunner

class Runner(BaseRunner):
    # nominal neXtSIM mesh resolution
    resolution = 10000
    # minimal a/p ratio
    r_min = 0.13
    # distance from RGPS nodes to neXtSIM nodes for initial subset
    distance_upper_bound1 = 100000
    cores = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # maximal area
        self.a_max = 2. * self.resolution ** 2
        # distance from RGPS elements to neXtSIM elements for final subset
        self.distance_upper_bound2 = self.resolution * 1.5


    def __call__(self, rfile, idir, ofile):
        if self.skip_processing(ofile): return ofile
        with np.load(rfile, allow_pickle=True) as ds:
            r_pairs = ds['pairs']
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