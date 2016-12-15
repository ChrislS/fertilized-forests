#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 15:05:00 2014

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

# Prepare...
n_classes = 3
n_trees = 200
ploty = [-6, 6, 100]
plotx = [-6, 6, 100]
X, Y = make_spiral(n_arms=n_classes, noise=.4)

#%%
soil = f.Soil('float', 'float', 'uint', f.Result_Types.probabilities)
for depth in [3, 6, 15]:
  cls = []
  lm = []
  for i in range(n_trees):
    cls.append(soil.ThresholdDecider(
                  soil.StandardFeatureSelectionProvider(1, 2, 2, 2),
#                  soil.AlignedSurfaceCalculator(),
#                  soil.LinearSurfaceCalculator(400, random_seed=1+i),
                  soil.QuadraticSurfaceCalculator(400,
                                                  np.array([X[:, 0].min(),
                                                            X[:, 0].max(),
                                                            X[:, 1].min(),
                                                            X[:, 1].max()], dtype='float'),
                                                  random_seed=1+i),
                  soil.ClassificationThresholdOptimizer(True,
                                                        n_classes,
                                                        soil.EntropyGain(soil.ShannonEntropy()))))
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
    plt.title('Depth %d' %(depth))
    point_prob_plot(forest, X, Y, plotx, ploty)
    plt.savefig('effect_depth_%d.png' % (depth))
    plt.show()
