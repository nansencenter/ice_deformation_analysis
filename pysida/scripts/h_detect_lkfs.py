#!/usr/bin/env python
import sys

import numpy as np
from multiprocessing import Pool
import pandas as pd

from pysida.lib import PairFilter, BaseRunner
from pysida.liblkf import Rasterizer, LKFDetector

class Runner(BaseRunner):
    params = dict(
        # minimum number of days between image pair
        # lower value allows more matching pairs but increases the tail of PDF
        min_time_diff = 0.5,
        # maximum number of days between image pair
        # higher value allows more matching images but increases range of values
        max_time_diff = 7.5,
        # aggregation window (days)
        window = '4D',
        # minimum area of triangle element to select for computing deformation
        # higher value decrease the number of elements but may exclude some
        # errorneous triangles appearing between RS2 swaths
        min_area = 0,
        # maximum area of triangle element to select for computing deformation
        # lower value decrease the number of elements but may exclude some
        # errorneous triangles appearing between RS2 swaths
        max_area = 300e6,
        # minimum area/perimeter ratio to select for computing deformation
        # lower value allow sharper angles in triangles and increases number of elements
        min_ap_ratio = 0.12,
        # minimum distance from coast
        min_dist=0,
        # minimum displacement between nodes
        min_drift=0,
        # minimum size of triangulation
        # lower number increases number of triangles but increase chance of noise
        min_tri_size = 10,
    )
    cores = 5

    def __call__(self, pfile, date_begin, date_end, freq, resolution):
        dates = pd.date_range(date_begin, date_end, freq=freq).to_pydatetime()

        dfile = pfile.replace('pairs.npz', 'defor.npz')
        ofile = pfile.replace('pairs.npz', 'lkfs.npz')
        print('Load', pfile)
        with np.load(pfile, allow_pickle=True) as f:
            pairs = f['pairs']
        print('Load', dfile)
        with np.load(dfile, allow_pickle=True) as f:
            defor = f['defor']

        pf = PairFilter(pairs, defor, resolution, **self.params)
        r = Rasterizer()
        detector = LKFDetector(pf, r)

        lkfs_defs = []
        with Pool(self.cores) as p:
            lkfs_defs = p.map(detector.proc_one_date, dates)

        print(ofile)
        np.savez(ofile, dates=dates, lkfs_defs=np.array(lkfs_defs, dtype="object"))
        return ofile


if __name__ == '__main__':
    pfile = str(sys.argv[1])
    date_begin = pd.Timestamp(sys.argv[2]) #2006-12-05
    date_end = pd.Timestamp(sys.argv[3]) # 2007-05-15
    # frequency at which to compute deformation snapshots, e.g. '1D'
    freq = str(sys.argv[4])
    # nominal mesh resolution
    resolution = int(sys.argv[5]) # 10000
    Runner()(pfile, date_begin, date_end, freq, resolution)