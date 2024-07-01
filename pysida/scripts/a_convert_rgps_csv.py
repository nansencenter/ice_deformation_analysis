#!/usr/bin/env python
import sys
from pysida.lib import dataframe_from_csv, BaseRunner

class Runner(BaseRunner):
    def __call__(self, ifile):
        ofile = ifile.replace('.csv', '.df')
        if self.skip_processing(ofile): return ofile
        df = dataframe_from_csv(ifile)
        print(ofile)
        df.to_pickle(ofile)
        return ofile

if __name__ == '__main__':
    ifile = str(sys.argv[1])
    Runner()(ifile)
