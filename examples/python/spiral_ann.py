#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 06 19:12:31 2014

@author: lassnech
"""

import theano
from pynnet import SimpleNode
import pynnet.nlins
from pynnet.nodes import errors
from pynnet.training import get_updates

import matplotlib.pyplot as plt
import numpy as np
from plottools import make_spiral, point_prob_plot, probaproxy
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
np.random.seed(1)

#%% Prepare...
n_classes = 2
ploty = [-6, 6, 100]
plotx = [-6, 6, 100]
X, Y = make_spiral(n_arms=n_classes, noise=.4)

#%% Build ANN.
sx = theano.tensor.matrix('x')
sy = theano.tensor.matrix('y')

h = SimpleNode(sx, 2, 4)
h2 = SimpleNode(h, 4, 2)
out = SimpleNode(h2, 2, 1, nlin=pynnet.nlins.sigmoid)
cost = errors.mse(out, sy)

theano.config.blas.ldflags=''
eval = theano.function([sx], out.output)
test = theano.function([sx, sy], cost.output)
train = theano.function([sx, sy], cost.output,
                        updates=get_updates(cost.params, cost.output, 0.01))

print("Error at start:", test(X, Y))

for i in range(200000):
    train(X, Y)
print("Error after 200000:", test(X, Y))

def pfunc(x):
  return 1. - eval(x)
clp = probaproxy(pfunc)

plt.figure()
point_prob_plot(clp, X, Y, plotx, ploty)
plt.title('ANN (2 hidden layers, 4, 2)')
plt.savefig('spiral_ann.png')
plt.show()