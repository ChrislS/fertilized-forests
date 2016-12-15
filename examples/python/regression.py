#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:11:37 2014

@author: Moritz Einfalt
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
from plottools import mean_squared_error
# For plotting and evaluation.
if INTERACTIVE:
    import matplotlib.pyplot as plt
np.random.seed(1)

# generate 2D samples from a polynomial function
n_samples = 45
annot_dim = 1
input_dim = 1
samplesX = np.random.uniform(low=-6,
                             high=6,
                             size=(n_samples, input_dim)).astype('float32')
samplesY = (samplesX ** 2) - 0.2 * samplesX - 1
noise_var = 2.3
random_noise = np.random.randn(n_samples, annot_dim) * noise_var
samplesY += random_noise

# define the plotted space
minX = samplesX.min()
maxX = samplesX.max()
deltaX = (maxX - minX)
plotX = [minX -0.2*deltaX, maxX +0.2*deltaX, 50]

# create and fit a float regression forest
soil = Soil('f', 'f', 'f', Result_Types.regression)
n_trees = 5
max_depth = 2
n_thresholds = 2
forest = soil.StandardRegressionForest(input_dim,
                                       max_depth,
                                       1,
                                       n_thresholds,
                                       n_trees,
                                       3,
                                       6)
forest.fit(samplesX, samplesY)

# predict the whole interval (minX, maxX)
predictX = np.linspace(plotX[0], plotX[1], num=plotX[2]).astype('float32')
predictX = np.atleast_2d(predictX).T
forest_result = forest.predict(predictX)
# the forest result is of shape [n_samples, 2*predict_dim]
# it contains for each sample the predictions, and the confidence per
# prediction dimension.
n_predictions = forest_result.shape[0]
mean_prediction = forest_result[:, :annot_dim]

# plot the result
if INTERACTIVE:
    plt.figure()
    plt.plot(predictX, mean_prediction, c='g', linewidth=1, label='Regression model')
    plt.plot(samplesX, samplesY ,linestyle='None', marker='o',
             markeredgecolor='blue', markeredgewidth=1,
             markerfacecolor='None', markersize=6,
             label='Samples')

    frame = plt.gca()
    frame.axes.get_xaxis().tick_bottom()
    frame.axes.get_yaxis().tick_left()
    plt.legend(loc=0)
    plt.xlabel("X", fontsize=18)
    plt.ylabel("Y", fontsize=18)
    plt.grid(True)
    plt.savefig('regression.png')
    plt.show()

assert mean_squared_error(samplesY, forest.predict(samplesX)[:, 0]) < 5.
