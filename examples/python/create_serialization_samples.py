#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:11:37 2014

@author: Christoph Lassner
"""
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))

import numpy as np
import fertilized
soil = fertilized.Soil('d', 'd', 'd', fertilized.Result_Types.regression)

tree = soil.StandardRegressionTree(1, 0, 1, 5, 2, 4)

X = np.array([[0.], [0.]])
Y = np.array([[0.], [10**3]])
tree.fit(X, Y)

print('Tree: %f, STD: %f' % (np.sqrt(tree.predict(np.array([[0.]]))[0,1]), np.std(Y[:, 0])))

if os.name == 'nt':
    tree.save('tree_windows.ft')
else:
    tree.save('tree_linux.ft')

