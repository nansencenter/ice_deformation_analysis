#!/usr/bin/env python
import sys

import numpy as np

from pysida.liblkf import get_average_lkf_stats

# path to LKFs file
lfile = str(sys.argv[1])
ofile = lfile.replace('lkfs.npz', 'lkf_stats.npz')
with np.load(lfile, allow_pickle=True) as f:
    dates = f['dates']
    lkfs_defs = f['lkfs_defs']

avg_lkf_stats = get_average_lkf_stats(lkfs_defs, dates)

print('Save', ofile)
np.savez(ofile, dates=dates, lkf_stats=avg_lkf_stats)
