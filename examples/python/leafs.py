#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 17:40:55 2014

@author: lassnech
"""

from sympy import plot, log, re, sqrt, pi, exp, Max
from sympy.abc import x
from sympy.utilities.lambdify import lambdify
import numpy as np

def gaussian(mu, sigma):
  return 1/sqrt(2*pi*sigma) * exp(-(x - mu) ** 2 / sigma ** 2)

rnge = (x, 0, 12)
g1 = gaussian(1, 15)
g2 = gaussian(11, 17)
g3 = gaussian(4, 0.2)
g4 = gaussian(8, 1)

def basic_plot():
  p1=plot(g1, rnge, show=False, line_color='b', ylim=(0., 1.))
  p2=plot(g2, rnge, show=False, line_color='g')
  p3=plot(g3, rnge, show=False, line_color='y')
  p4=plot(g4, rnge, show=False, line_color='r')

  p1.extend(p2)
  p1.extend(p3)
  p1.extend(p4)
  return p1

p1 = basic_plot()
p1.title = 'sample result distributions'
p1.save('dist_basic.png')
p1.show()

means = 1./4.*(g1+g2+g3+g4)
p1.title = 'averaging'
p1.extend(plot(means, rnge, show=False, line_color='black'))
p1.save('dist_means.png')
p1.show()


p1 = basic_plot()
p1.title = 'multiplication'
mult = g1*g2*g3*g4
# Find pseudo max of mult.
multf = lambdify(x, mult)
xv = np.linspace(0, 12, 1000)
sumv = 0.
maxv = 0.
for val in xv:
  v = multf(val)
  sumv += v
  if (v > maxv):
    maxv = v
p1.extend(plot(1./maxv * mult, rnge, show=False, line_color='black'))
p1.save('dist_mult.png')
p1.show()
