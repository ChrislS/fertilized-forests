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

# Generate a soil to produce all succeeding forest elements.
# It ensures datatype consistency.
soil = Soil()
# For a custom soil, you could use for example:
#soil = f.Soil('uint8',
#              'int16',
#              'uint',
#              f.Result_Types.hough_map,
#              f.Result_Types.hough_map)

# A standard classifier can now be created easily:
tree = soil.StandardClassificationTree(
                2, # number of classes
                2) # number of features
# Fit the tree on the data (one sample per row).
tree.fit(X, Y)
assert accuracy_score(Y, np.argmax(tree.predict(X), axis=1)) > 0.95

# Plot the results.
if INTERACTIVE:
  plt.figure()
  point_prob_plot(tree, X, Y, plotx, ploty)
  plt.savefig('overview_tree.png')
  plt.show()

###############################################################################
# Do this with a forest:
forest = soil.StandardClassificationForest(2, 2)
# This constructs a forest of ten trees!
forest.fit(X, Y)
assert accuracy_score(Y, np.argmax(forest.predict(X), axis=1)) > 0.95
if INTERACTIVE:
  plt.figure()
  point_prob_plot(forest, X, Y, plotx, ploty)
  plt.savefig('overview_forest.png')
  plt.show()
