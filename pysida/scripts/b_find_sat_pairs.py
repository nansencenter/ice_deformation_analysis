#!/usr/bin/env python
from datetime import datetime
import sys

import pandas as pd
import numpy as np

from pysida.lib import GetSatPairs, BaseRunner

class Runner(BaseRunner):
    def __call__(self, ifile, date0, date1):
        ofile = ifile.replace('_LP.df', f'_{date1.strftime("%Y%m%d")}_pairs.npz')
        if self.skip_processing(ofile): return ofile
        df = pd.read_pickle(ifile)
        gsp = GetSatPairs(df, **self.__dict__)
        pairs = gsp.get_all_pairs(date0, date1)
        print('N pairs', len(pairs), ofile)
        np.savez(ofile, pairs=pairs)
        return ofile

if __name__ == '__main__':
    ifile = str(sys.argv[1])
    date0 = datetime.strptime(str(sys.argv[2]), '%Y-%m-%d')
    date1 = datetime.strptime(str(sys.argv[3]), '%Y-%m-%d')
    Runner()(ifile, date0, date1)