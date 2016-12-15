# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 11:42:42 2014

@author: lassnech
"""

import numpy as np


def accuracy_score(gt, pred):
    return np.sum(gt.flat == pred.flat) / float(len(gt.flat))


def mean_squared_error(gt, pred):
    return np.mean(np.power(gt.flat[:] - pred.flat[:], 2))


def to_im_coords(points, im_spec):
  retpoints = []
  for point in points:
    if point < im_spec[0] or point > im_spec[1]:
      print('Warning: detected point outside im region.')
      continue
    retpoints.append((point - im_spec[0]) / (im_spec[1] - im_spec[0]) * im_spec[2])
  return np.array(retpoints)

def point_prob_plot(classifier, X, Y, plotx, ploty):
  import matplotlib
  import matplotlib.pyplot as plt
  n_classes = np.unique(Y).shape[0]
  xmb = np.linspace(plotx[0], plotx[1], num=plotx[2])
  ymb = np.linspace(ploty[0], ploty[1], num=ploty[2])
  Xm, Ym = np.meshgrid(xmb, ymb)
  matplotlib.rcParams['xtick.direction'] = 'out'
  matplotlib.rcParams['ytick.direction'] = 'out'

  coltemp = np.array([0,0,0], dtype='uint8')
  Zm = np.zeros((Xm.shape[0], Xm.shape[1], 3), dtype='uint8')
  for y in range(Zm.shape[0]):
    for x in range(Zm.shape[1]):
      prediction = classifier.predict(np.array([[Xm[y, x], Ym[y, x]]], dtype='float32'))
      coltemp[2] = (prediction[0,0] * 255.).astype('uint8')
      if n_classes == 2:
        if prediction.ndim == 2 and prediction.shape[1] == 2:
          coltemp[0] = (prediction[0,1] * 255.).astype('uint8')
        else:
          coltemp[0] = ((1. - prediction[0,0]) * 255.).astype('uint8')
      elif n_classes > 2:
        coltemp[0] = (prediction[0,2] * 255.).astype('uint8')
        coltemp[1] = (prediction[0,1] * 255.).astype('uint8')
      Zm[y, x] = coltemp.copy()
  plt.imshow(Zm, origin='lower')
  Xtransformed = to_im_coords(X[:, 0], plotx)
  Ytransformed = to_im_coords(X[:, 1], ploty)
  plt.scatter(Xtransformed, Ytransformed, c=Y)
  # Fix ticks.
  frame = plt.gca()
  frame.axes.get_xaxis().set_visible(False)
  frame.axes.get_yaxis().set_visible(False)

def point_on_circle(phi, r):
  return r * np.array([np.cos(phi), np.sin(phi)])

def make_spiral(n_samples_per_arm = 100, n_arms=2, noise=0.1):
  starting_angles = np.arange(n_arms) * 2. * np.pi / float(n_arms)
  points = np.empty((n_arms * n_samples_per_arm, 2), dtype='float32')
  ids = np.empty((points.shape[0], 1), dtype='uint32')
  maxpifac = 1.7
  for arm_id in range(n_arms):
    angle = starting_angles[arm_id]
    for point_id in range(n_samples_per_arm):
      angle_add = np.random.uniform(low=0., high=maxpifac)
      position = point_on_circle(angle + angle_add * np.pi, 1. + 2. * angle_add)
      position += np.random.normal(scale=noise * angle_add, size=(2))
      points[arm_id * n_samples_per_arm + point_id] = position
      ids[arm_id * n_samples_per_arm + point_id] = arm_id
  return points, ids


class probaproxy:
  def __init__(self, clffunc):
    self._clf = clffunc

  def predict(self, point):
    return self._clf(point)
