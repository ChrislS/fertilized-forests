#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 11:23:41 2014

@author: lassnech
"""

import sympy
from sympy.abc import x as sx, y as sy
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

rw = np.matrix([sx, sy, 1])

#%%
x = np.linspace(-200, 200, num=100)
y = np.linspace(-200, 200, num=100)
X, Y = np.meshgrid(x, y)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


#%%
Q = np.matrix(np.random.uniform(low=-1., high=1., size=(3, 3)))
Q /= np.linalg.norm(Q)
#Q = np.matrix(np.random.normal(size=(2, 2)))
#P = np.matrix(np.random.normal(size=(1, 2)))
eq = rw * (Q * rw.T) #+ P*rw.T
eqsy = sympy.solve(eq[0,0], sy)
sympy.plot(*[(teq, (sx, -1, 1)) for teq in eqsy])

#%%
Z = np.zeros(X.shape)
for y in range(Z.shape[0]):
  for x in range(Z.shape[1]):
    posvec = np.matrix([Y[y, x], X[y, x], 1.])
    Z[y, x] = posvec*(Q * posvec.T) #+ P*posvec.T
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Quadric contours')

