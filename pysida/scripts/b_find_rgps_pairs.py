#!/usr/bin/env python
from datetime import datetime
import sys

import pandas as pd
import numpy as np

from pysida.lib import get_rgps_pairs, BaseRunner

class Runner(BaseRunner):
    min_time_diff = 0.5
    max_time_diff = 7
    min_size = 100
    r_min = 0.12
    a_max = 200e6
    cores = 5

    def __call__(self, ifile, date0, date1):
        ofile = ifile.replace('_LP.df', '_pairs.npz')
        df = pd.read_pickle(ifile)
        pairs = get_rgps_pairs(
            df, date0, date1,
            min_time_diff=self.min_time_diff,
            max_time_diff=self.max_time_diff,
            min_size=self.min_size,
            r_min=self.r_min,
            a_max=self.a_max,
            cores=self.cores
        )
        print('N pairs', len(pairs), ofile)
        np.savez(ofile, pairs=pairs)
        return ofile

if __name__ == '__main__':
    ifile = str(sys.argv[1])
    date0 = datetime.strptime(str(sys.argv[2]), '%Y-%m-%d')
    date1 = datetime.strptime(str(sys.argv[3]), '%Y-%m-%d')
    Runner()(ifile, date0, date1)