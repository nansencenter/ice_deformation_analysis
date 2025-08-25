#!/usr/bin/env python
from multiprocessing import Pool
import sys

import numpy as np
import pandas as pd

from pysida.lib import PairFilter, MarsanSpatialScaling, BaseRunner

class Runner(BaseRunner):
    cores = 4
    def __call__(self, pfile, date_begin, date_end, freq):
        dfile = pfile.replace('pairs.npz', 'defor.npz')
        ofile = pfile.replace('pairs.npz', 'scale.npz')
        if self.skip_processing(ofile): return ofile
        with np.load(pfile, allow_pickle=True) as f:
            pairs = list(f['pairs'])
        with np.load(dfile, allow_pickle=True) as f:
            defor = list(f['defor'])

        pf = PairFilter(pairs, defor, dist2coast_path=self.dist2coast_path, resolution=self.resolution)
        mss = MarsanSpatialScaling(pf)
        dates = pd.date_range(date_begin, date_end, freq=freq).to_pydatetime()
        if self.cores <= 1:
            mmm = [mss.proc_one_date(date) for date in dates]
        else:
            with Pool(self.cores) as p:
                mmm = p.map(mss.proc_one_date, dates)
        print(ofile)
        np.savez(ofile, dates=dates, mmm=mmm)
        return ofile


if __name__ == '__main__':
    pfile = str(sys.argv[1])
    date_begin = pd.Timestamp(sys.argv[2])
    date_end = pd.Timestamp(sys.argv[3])
    freq = str(sys.argv[4])
    resolution = int(sys.argv[5])
    Runner()(pfile, date_begin, date_end, freq, resolution)