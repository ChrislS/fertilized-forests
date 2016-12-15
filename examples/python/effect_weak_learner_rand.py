#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 15:29:28 2014

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
from plottools import make_spiral, point_prob_plot, accuracy_score
np.random.seed(1)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#%% Prepare...
n_classes = 3
n_trees = 400
ploty = [-6, 6, 100]
plotx = [-6, 6, 100]
X, Y = make_spiral(n_arms=n_classes, noise=.4)

#%%
soil = f.Soil('float', 'float', 'uint', f.Result_Types.probabilities)
depth = 13
rs=1
for suggestions in [3, 250]:
  for dec_name in ['aligned', 'linear', 'quadratic']:
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
      cls.append(soil.ThresholdDecider(   feat_sel_prov,
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
    forest.fit(X, Y, 1)
    assert accuracy_score(Y, np.argmax(forest.predict(X), axis=1)) > 0.95

    if INTERACTIVE:
        plt.figure()
        plt.title('%s decider, $|\Theta_j|=%d$' %(dec_name, suggestions * 2))
        point_prob_plot(forest, X, Y, plotx, ploty)
        plt.savefig('effect_wl%s_rand_%d.png' % (dec_name, suggestions * 2))
        plt.show()
