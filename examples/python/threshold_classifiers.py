#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 10:56:16 2014

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

import numpy as np
if INTERACTIVE:
    import matplotlib.pyplot as plt

import fertilized as f

np.random.seed(2)
# #%% Prepare.
# from sklearn.datasets import make_classification
# X, Y = make_classification(n_features=2,
#                            n_redundant=0,
#                            n_repeated=0,
#                            n_classes=4,
#                            n_clusters_per_class=1)
# X = X.astype("float32")
# Y = np.atleast_2d(Y.astype("uint32")).T
# np.savetxt('toydata2_x.csv', X)
# np.savetxt('toydata2_y.csv', Y)
X = np.genfromtxt('toydata2_x.csv').astype('float32')
Y = np.atleast_2d(np.genfromtxt('toydata2_y.csv').astype('uint32')).T
soil = f.Soil()
xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
ys = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
Xm, Ym = np.meshgrid(xs, ys)

for dec_name in ['aligned', 'linear', 'quadratic']:
  #%% Create stump.
  if dec_name == 'aligned':
    feat_sel_prov = soil.StandardFeatureSelectionProvider(2, 1, 2, 2)
  else:
    feat_sel_prov = soil.StandardFeatureSelectionProvider(1, 2, 2, 2)
  if dec_name == 'aligned':
    feat_calc = soil.AlignedSurfaceCalculator()
  elif dec_name == 'linear':
    feat_calc = soil.LinearSurfaceCalculator(400)
  elif dec_name == 'quadratic':
    feat_calc = soil.QuadraticSurfaceCalculator(400,
                                                np.array([X[:, 0].min(),
                                                          X[:, 0].max(),
                                                          X[:, 1].min(),
                                                          X[:, 1].max()],
                                                        dtype='float32'))
  stump = soil.Tree(1,
                    1,
                    2,
                    soil.ThresholdDecider(
                      feat_sel_prov,
                      feat_calc,
                      soil.ClassificationThresholdOptimizer(
                        True,
                        4,
                        soil.EntropyGain(soil.InducedEntropy(2))
                      ),
                    ),
                    soil.ClassificationLeafManager(4)
                    )
  stump.fit(X, Y)

  #%% Plot.
  Zm = stump.predict(np.ascontiguousarray(
                       np.vstack((Xm.ravel(), Ym.ravel())).T.astype('float32')))[:, 0].reshape(100, 100)
  if INTERACTIVE:
      plt.figure()
      plt.contour(Xm, Ym, Zm)
      plt.scatter(X[:, 0], X[:, 1], c=Y)
      plt.title(dec_name)
      plt.savefig('decider_%s.png' % (dec_name))
      plt.show()
