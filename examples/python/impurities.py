#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 05 14:04:50 2014

@author: lassnech
"""

from sympy import plot, log, re, Abs
from sympy.abc import x

sh_entropy = re(-x*log(x)/log(2)-(1-x)*log(1-x)/log(2))
ind_p_1 = 1-Abs(0.5-x)**1 * 2
ind_p_2 = 1-Abs(0.5-x)**2 * 4
ind_p_3 = 1-Abs(0.5-x)**3 * 8
ind_p_4 = 1-Abs(0.5-x)**4 * 16
rnge = (x, 0, 1)
p1 = plot(sh_entropy, rnge, show=False, line_color='b')
p2 = plot(ind_p_2, rnge, show=False, line_color='g')
p3 = plot(ind_p_3, rnge, show=False, line_color='y')
p4 = plot(ind_p_4, rnge, show=False, line_color='r')
p5 = plot(ind_p_1, rnge, show=False, line_color='purple')

p1.extend(p2)
p1.extend(p3)
p1.extend(p4)
p1.extend(p5)
p1.title = 'entropies'
p1.save('impurities.png')
p1.show()
