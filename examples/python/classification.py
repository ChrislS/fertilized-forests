#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))
if len(sys.argv) > 1:
  print('Quiet mode. Plot display disabled.')
  INTERACTIVE = False
else:
  INTERACTIVE = True

import numpy as np

import fertilized as f
if INTERACTIVE:
    import matplotlib.pyplot as plt
from plottools import point_prob_plot, make_spiral, accuracy_score

np.random.seed(1)

ploty = [-8, 8, 100]
plotx = [-8, 8, 100]

#%% Prepare.
# from sklearn.datasets import make_classification
# X, Y = make_classification(n_features=2,
#                            n_redundant=0,
#                            n_repeated=0,
#                            n_classes=2,
#                            n_clusters_per_class=1,
#                            class_sep=5.,
#                            flip_y=0.,
#                            random_state=6)
# X = X.astype("float32")
# Y = np.atleast_2d(Y.astype("uint32")).T
# np.savetxt('toydata_x.csv', X)
# np.savetxt('toydata_y.csv', Y)
X = np.genfromtxt('toydata_x.csv').astype('float32')
Y = np.atleast_2d(np.genfromtxt('toydata_y.csv').astype('uint32')).T

if INTERACTIVE:
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

soil = f.Soil()
xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
ys = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
Xm, Ym = np.meshgrid(xs, ys)

stumps = []
for rs in range(1, 201):
  dec_name='aligned'
  #%% Create stump.
  if dec_name == 'aligned':
    feat_sel_prov = soil.StandardFeatureSelectionProvider(2, 1, 2, 2,
                                                          random_seed=rs)
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
                      soil.RandomizedClassificationThresholdOptimizer(
                        2,
                        2,
                        soil.EntropyGain(soil.InducedEntropy(2)),
                        random_seed=rs
                      ),
                    ),
                    soil.ClassificationLeafManager(2)
                   )

  stump.fit(X, Y)
  stumps.append(stump)

  #%% Plot.
  if rs in [1, 2]:
    Zm = stump.predict(np.ascontiguousarray(
                         np.vstack((Xm.ravel(), Ym.ravel())).T.astype('float32')))[:, 0].reshape(100, 100)
    if INTERACTIVE:
        plt.figure()
        plt.contour(Xm, Ym, Zm)
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.xlim((plotx[0], plotx[1]))
        plt.ylim((ploty[0], ploty[1]))
        plt.gca().set_aspect('equal')
        plt.title('Tree %d' % (rs))
        plt.savefig('classification_stump_%d.png' % (rs))
        plt.show()

print("Stump 0 depth: ", stumps[0].depth())
forest8 = soil.CombineTrees(stumps[:8])
forest = soil.CombineTrees(stumps)

if INTERACTIVE:
    plt.figure()
    point_prob_plot(forest8, X, Y, plotx, ploty)
    plt.title('8 Trees')
    plt.savefig('uncertainty_quality_8.png')

    plt.figure()
    point_prob_plot(forest, X, Y, plotx, ploty)
    plt.title('200 Trees')
    plt.savefig('uncertainty_quality_200.png')
assert accuracy_score(Y, np.argmax(forest.predict(X), axis=1)) > 0.95
