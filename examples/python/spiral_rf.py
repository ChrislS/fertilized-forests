#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 20:22:41 2014

@author: lassnech
"""
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))
if len(sys.argv) > 1:
  print('Quiet mode. Plot display disabled.')
  INTERACTIVE = False
else:
  INTERACTIVE = True

import fertilized as f
if INTERACTIVE:
    import matplotlib.pyplot as plt

import numpy as np
np.random.seed(8)
from plottools import make_spiral, point_prob_plot, probaproxy, accuracy_score
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
np.random.seed(1)

#%% Prepare...
n_classes = 2
n_trees = 200
ploty = [-6, 6, 100]
plotx = [-6, 6, 100]
X, Y = make_spiral(n_arms=n_classes, noise=.4)


# Forest.
soil = f.Soil()
dec_name = 'quadratic'
rs = 1
suggestions = 5
depth = 5
if dec_name == 'aligned':
  feat_sel_prov = soil.StandardFeatureSelectionProvider(2, 1, 2, 2, random_seed=rs)
else:
  feat_sel_prov = soil.StandardFeatureSelectionProvider(1, 2, 2, 2)
if dec_name == 'aligned':
  feat_calc = soil.AlignedSurfaceCalculator()
elif dec_name == 'linear':
  feat_calc = soil.LinearSurfaceCalculator(2, random_seed=rs)
elif dec_name == 'quadratic':
  feat_calc = soil.QuadraticSurfaceCalculator(2,
                                              np.array([X[:, 0].min(),
                                                        X[:, 0].max(),
                                                        X[:, 1].min(),
                                                        X[:, 1].max()],
                                                      dtype='float32'),
                                              random_seed=rs)
cls = []
lm = []
for i in range(n_trees):
  cls.append(soil.ThresholdDecider(feat_sel_prov,
                                      feat_calc,
                                      soil.RandomizedClassificationThresholdOptimizer(
                                        suggestions,
                                        n_classes,
                                        soil.EntropyGain(soil.ShannonEntropy()),
                                        random_seed=rs)))
  lm.append(soil.ClassificationLeafManager(n_classes))
forest = soil.Forest(depth,
                     1,
                     2,
                     n_trees,
                     cls,
                     lm,
                     soil.ClassicTraining(soil.NoBagging()))
forest.fit(X, Y)

if INTERACTIVE:
    plt.figure()
    point_prob_plot(forest, X, Y, plotx, ploty)
    plt.title('Random Forest $(T=%d, D=%d, |\Theta_j|=%d)$' % (n_trees, depth, suggestions * 2))
    plt.savefig('spiral_rf.png')
    plt.show()
assert accuracy_score(Y, np.argmax(forest.predict(X), axis=1)) > 0.95
