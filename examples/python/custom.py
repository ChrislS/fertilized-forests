#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 07 14:55:11 2014

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

from fertilized import *
import numpy as np
from plottools import make_spiral, point_prob_plot, accuracy_score

# Only for plotting, evaluation.
if INTERACTIVE:
    import matplotlib.pyplot as plt

np.random.seed(8)

# Generate the spiral dataset for further use.
X, Y = make_spiral()
plotx = [-6, 6, 100]
ploty = [-6, 6, 100]

if INTERACTIVE:
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

soil = Soil('f', 'f', 'uint', Result_Types.probabilities)

###############################################################################
# Lets build a customized forest:
depth = 6
n_trees = 200
# These variables will contain the classifiers and leaf managers for each tree.
cls = []
lm = []
for i in range(n_trees):
  # Classifier:
  cls.append(soil.ThresholdDecider(
               soil.StandardFeatureSelectionProvider(1, 2, 2, 2, random_seed=1+i),
               soil.LinearSurfaceCalculator(400, random_seed=1+i),
               soil.RandomizedClassificationThresholdOptimizer(
                 True,
                 2,
                 soil.EntropyGain(soil.ShannonEntropy()), random_seed=1+i)))
  # Leaf manager:
  lm.append(soil.ClassificationLeafManager(2))

forest = soil.Forest(depth,
                     1,
                     2,
                     n_trees,
                     cls,
                     lm,
                     soil.ClassicTraining(soil.NoBagging()))
forest.fit(X, Y)
assert accuracy_score(Y, np.argmax(forest.predict(X), axis=1)) > 0.95
if INTERACTIVE:
  plt.figure()
  point_prob_plot(forest, X, Y, plotx, ploty)
  plt.savefig('custom.png')
  plt.show()
