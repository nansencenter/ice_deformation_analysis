#!/usr/bin/env python

import sys

from ida.lib import dataframe_from_csv

ifile = str(sys.argv[1])
ofile = ifile.replace('.csv', '.df')
print(ofile)

df = dataframe_from_csv(ifile)
df.to_pickle(ofile)