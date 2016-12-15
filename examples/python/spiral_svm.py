#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 18:42:59 2014

@author: lassnech
"""

from sklearn import svm
from sklearn import grid_search
import matplotlib.pylab as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from plottools import make_spiral, point_prob_plot, probaproxy
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

np.random.seed(1)

n_classes = 2
n_trees = 200
ploty = [-6, 6, 100]
plotx = [-6, 6, 100]

X, Y = make_spiral(n_arms=n_classes, noise=.4)

##############################################################################
parameters = {'kernel': ['rbf'],
              'C': [1, 10, 100, 1000, 10000, 100000],
              'gamma': [10 ** x for x in range(-5, 3)],
              'probability':[True]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X, Y.ravel())

svmp = probaproxy(clf.predict_proba)

plt.figure()
point_prob_plot(svmp, X, Y, plotx, ploty)
plt.title('RBF kernel SVM $(\gamma=%d, C=%f)$' % (clf.best_params_['gamma'], clf.best_params_['C']))
plt.savefig('spiral_svm.png')
plt.show()

##############################################################################
parameters = {'C': [1, 10, 100, 1000, 10000],
              'probability': [True]}
svr = svm.SVC(kernel='linear', max_iter=20000)
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X, Y.ravel())

svmp = probaproxy(clf.predict_proba)

plt.figure()
point_prob_plot(svmp, X, Y, plotx, ploty)
plt.title('Linear SVM $(C=%d)$' % (clf.best_params_['C']))
plt.savefig('spiral_svm_linear.png')
plt.show()
