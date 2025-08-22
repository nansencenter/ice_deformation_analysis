#!/usr/bin/env python
import sys

import numpy as np

from pysida.lib import MeshFileList, collocate_pairs, BaseRunner

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


    def __call__(self, src_file, src_dir, dst_file_sat, dst_file_nextsim):
        if self.skip_processing(dst_file_sat): return dst_file_sat, dst_file_nextsim
        with np.load(src_file, allow_pickle=True) as ds:
            sat_src_pairs = ds['pairs']
        mfl = MeshFileList(src_dir, lazy=True)
        pairs = collocate_pairs(
            sat_src_pairs,
            mfl,
            self.r_min,
            self.a_max,
            self.distance_upper_bound1,
            cores=self.cores
        )
        sat_pairs, nextsim_pairs = [], []
        for sat_pair_nextsim_pair in pairs:
            if sat_pair_nextsim_pair and len(sat_pair_nextsim_pair) == 2:
                sat_pairs.append(sat_pair_nextsim_pair[0])
                nextsim_pairs.append(sat_pair_nextsim_pair[1])
        print(dst_file_sat, 'N pairs:', len(sat_pairs))
        np.savez(dst_file_sat, pairs=sat_pairs)
        print(dst_file_nextsim, 'N pairs:', len(nextsim_pairs))
        np.savez(dst_file_nextsim, pairs=nextsim_pairs)
        return dst_file_sat, dst_file_nextsim

if __name__ == '__main__':
    src_file = str(sys.argv[1])
    src_dir = str(sys.argv[2])
    dst_file_sat = str(sys.argv[3])
    dst_file_nextsim = str(sys.argv[4])
    Runner()(src_file, src_dir, dst_file_sat, dst_file_nextsim)