#!/usr/bin/env python
import os
import sys
from pysida.lib import dataframe_from_csv, BaseRunner

class Runner(BaseRunner):
    def __call__(self, ifile, odir):
        ofile = ifile.replace('.csv', '.df')
        if odir is not None:
            ofile = os.path.join(odir, os.path.basename(ofile))
        if self.skip_processing(ofile): return ofile
        df = dataframe_from_csv(ifile)
        print(ofile)
        df.to_pickle(ofile)
        return ofile

if __name__ == '__main__':
    ifile = str(sys.argv[1])
    odir = str(sys.argv[2])
    Runner()(ifile, odir)
