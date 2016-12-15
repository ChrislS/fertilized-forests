#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Christoph Lassner
# Platform independent serialization.
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))

import numpy as np
from fertilized import *

soil = Soil('d', 'd', 'd', Result_Types.regression)
# Deserialize tree serialized on Windows.
try:
    tree_win = soil.TreeFromFile("tree_windows.ft")
except RuntimeError as re:
    if str(re).startswith("To use the libraries serialization features"):
        # All good, the library has been built without serialization support.
        sys.exit(0)
    else:
        raise Exception(str(re))
# Deserialize tree serialized on Linux.
tree_lin = soil.TreeFromFile("tree_linux.ft")

prediction = tree_win.predict(np.array([[0.]]))
assert prediction[0,0] == 500.
assert prediction[0,1] == 500000.

prediction = tree_lin.predict(np.array([[0.]]))
assert prediction[0,0] == 500., 'Expected 500.: %f' % (prediction[0,0])
assert prediction[0,1] == 500000.
