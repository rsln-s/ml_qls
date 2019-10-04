#!/usr/bin/env python

# Generates R-MAT random graphs using snap.py and saves them in Pajek format for later use
# Only works in Python 2

import os
import sys
import snap
import numpy as np

folder = '/zfs/safrolab/users/rshaydu/quantum/data/graphs/rmat/'

Rnd = snap.TRnd()

params = []
for nedges in [5000, 10000, 50000, 100000, 500000, 1000000]:
    for prob_a in np.linspace(0.4, 0.7, 7):
        params.append((nedges, prob_a, (0.8 - prob_a) / 2, (0.8 - prob_a) / 2))
        params.append((nedges, prob_a, (0.9 - prob_a) / 2, (0.75 - prob_a) / 2))

for param in params:
    outname = os.path.join(folder,
                           "_".join([str(x) for x in param] + ['rmat.out']))
    Graph = snap.GenRMat(1000, param[0], param[1], param[2], param[3], Rnd)
    print "Saving to %s" % outname
    snap.SavePajek(Graph, outname)
