#!/usr/bin/env python
from multiprocessing import Pool
import sys

import numpy as np

from pysida.lib import get_deformation_from_pair, BaseRunner


class Runner(BaseRunner):
    cores = 5
    def __call__(self, pfile):
        ofile = pfile.replace('_pairs.npz', '_defor.npz')
        if self.skip_processing(ofile): return ofile
        pairs = np.load(pfile, allow_pickle=True, fix_imports=False)['pairs']
        with Pool(self.cores) as p:
            defor = p.map(get_deformation_from_pair, pairs)
        print(ofile)
        np.savez(ofile, defor=np.array(defor, dtype='object'))
        return ofile

if __name__ == '__main__':
    pfile = str(sys.argv[1])
    Runner()(pfile)
