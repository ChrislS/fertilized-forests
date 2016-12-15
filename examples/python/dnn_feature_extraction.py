#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:40:48 2015

@author: christoph
"""
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))

import numpy as np
import numpy.testing as npt
import fertilized

###############################################################################
# Check the availability (since it's an optional module).
soil = fertilized.Soil()
try:
    extractor = soil.DNNFeatureExtractor(True) # CPU only.
except:
    print("The library has been built without caffe. Features cannot be extracted.")
    sys.exit(0)

###############################################################################
# Prepare the images.
from PIL import Image
timage_zeros = np.zeros((227, 227, 3), dtype='float32')
timage_ones = np.ones((227, 227, 3), dtype='float32')
timage_opencv = np.array(Image.open(os.path.join(os.path.dirname(__file__), 'sample-resized.png'))).astype('float32')
timage_opencv = np.ascontiguousarray(timage_opencv[:, :, ::-1])

# Assume the image has had its means subtracted already.
zero_res = extractor.extract([timage_zeros], False)
one_res = extractor.extract([timage_ones], False)
opencv_res = extractor.extract([timage_opencv], False)
# Load the caffe results.
zero_res_caffe = np.load(os.path.join(os.path.dirname(__file__), 'zero_res.npy'))
one_res_caffe = np.load(os.path.join(os.path.dirname(__file__), 'one_res.npy'))
opencv_res_caffe = np.load(os.path.join(os.path.dirname(__file__), 'opencv_res.npy'))
# Comparisons.
npt.assert_allclose(zero_res_caffe, zero_res, atol=1e-5)
npt.assert_allclose(one_res_caffe, one_res, atol=1e-5)
npt.assert_allclose(opencv_res_caffe, opencv_res, atol=1e-4)

# Test the order of results.
zero_one_res = extractor.extract([timage_zeros, timage_ones], False)
assert np.all(zero_one_res[0] == zero_res)
assert np.all(zero_one_res[1] == one_res)

# GPU
try:
    extractor = soil.DNNFeatureExtractor(False) # GPU
except:
    print("The library has been built without caffe GPU. Skipping.")
    sys.exit(0)
# Assume the image has had its means subtracted already.
zero_res = extractor.extract([timage_zeros], False)
one_res = extractor.extract([timage_ones], False)
opencv_res = extractor.extract([timage_opencv], False)
# Comparisons.
npt.assert_allclose(zero_res_caffe, zero_res, atol=1e-5)
npt.assert_allclose(one_res_caffe, one_res, atol=1e-5)
npt.assert_allclose(opencv_res_caffe, opencv_res, atol=1e-4)
# Order of results.
zero_one_res = extractor.extract([timage_zeros, timage_ones], False)
assert np.all(zero_one_res[0] == zero_res)
assert np.all(zero_one_res[1] == one_res)
