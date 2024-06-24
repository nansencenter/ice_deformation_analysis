#!/usr/bin/env python
import sys
from pysida.lib import dataframe_from_csv

def run(ifile):
    ofile = ifile.replace('.csv', '.df')
    print(ofile)
    df = dataframe_from_csv(ifile)
    df.to_pickle(ofile)

if __name__ == '__main__':
    ifile = str(sys.argv[1])
    run(ifile)
