#!/usr/bin/python

import scipy.io
import numpy as np
import sys
import os
import pdb

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print 'Usage: {0} <fn.mat>'.format(sys.argv[0])
        sys.exit()

    mat_fn = sys.argv[1]

    mat = scipy.io.loadmat(mat_fn)
    pdb.set_trace()


