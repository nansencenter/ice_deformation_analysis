#!/usr/bin/env python
import sys

import numpy as np
import pandas as pd

from pysida.lib import BaseRunner, MeshFileList, pair_from_nextsim_snapshots

class Runner(BaseRunner):
    r_min=0.12
    a_max=200e6

    def __call__(self, nextsim_dir, date_begin, date_end, freq, ofile):
        if self.skip_processing(ofile): return ofile
        mfl = MeshFileList(nextsim_dir)
        dates = pd.date_range(date_begin, date_end, freq=freq).to_pydatetime()
        pairs = []
        for i in range(len(dates) - 1):
            f0, d0 = mfl.find_nearest(dates[i])
            f1, d1 = mfl.find_nearest(dates[i+1])
            pair = pair_from_nextsim_snapshots(f0, f1, d0, d1, r_min=self.r_min, a_max=self.a_max)
            pairs.append(pair)
        print('N pairs', len(pairs), ofile)
        np.savez(ofile, pairs=pairs)
        return ofile

if __name__ == '__main__':
    nextsim_dir = str(sys.argv[1])
    date_begin = pd.Timestamp(sys.argv[2])
    date_end = pd.Timestamp(sys.argv[3])
    freq = str(sys.argv[4])
    ofile = str(sys.argv[5])
    Runner()(nextsim_dir, date_begin, date_end, freq, ofile)
