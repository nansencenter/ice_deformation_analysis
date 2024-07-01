#!/usr/bin/env python
import sys

import numpy as np

from pysida.lib import BaseRunner
from pysida.liblkf import get_average_lkf_stats

class Runner(BaseRunner):
    def __call__(self, lfile):
        ofile = lfile.replace('lkfs.npz', 'lkf_stats.npz')
        if self.skip_processing(ofile): return ofile
        with np.load(lfile, allow_pickle=True) as f:
            dates = f['dates']
            lkfs_defs = f['lkfs_defs']
        avg_lkf_stats = get_average_lkf_stats(lkfs_defs, dates)
        print(ofile)
        np.savez(ofile, lkf_stats=avg_lkf_stats)
        return ofile

if __name__ == '__main__':
    lfile = str(sys.argv[1])
    Runner()(lfile)
