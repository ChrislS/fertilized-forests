#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:39:48 2015

@author: Christoph Lassner
"""

import os
import numpy as np
import cv2
os.chdir('/home/christoph/git/caffe-brewer/python')
import caffe
TARGET_FOLDER = '/home/christoph/git/fertilized-forests/Presentation/Python'
net = caffe.Net(r'/home/christoph/caffe-models/bvlc_alexnet/alexnet_extraction.prototxt',
                r'/home/christoph/caffe-models/bvlc_alexnet/bvlc_alexnet.caffemodel', caffe.TEST)
timage_zeros = np.zeros((1, 3, 227, 227), dtype='float32')
timage_ones = np.ones((1, 3, 227, 227), dtype='float32')
timage_original = cv2.imread('%s/sample.png'%(TARGET_FOLDER))
timage_opencv = cv2.resize(timage_original, (227, 227))
cv2.imwrite('%s/sample-resized.png'%(TARGET_FOLDER), timage_opencv)
# Change axes order.
timage_opencv = timage_opencv.transpose((2, 0, 1))
# To blob
timage_opencv.shape = [1, 3, 227, 227]
# To correct type.
timage_opencv = np.ascontiguousarray(timage_opencv.astype('float32'))
zero_res = net.forward_all(blobs=['pool5'], **{net.inputs[0]: timage_zeros})['pool5']
one_res = net.forward_all(blobs=['pool5'], **{net.inputs[0]: timage_ones})['pool5']
opencv_res = net.forward_all(blobs=['pool5'], **{net.inputs[0]: timage_opencv})['pool5']
# Results
np.save('%s/zero_res.npy'%(TARGET_FOLDER), zero_res)
np.save('%s/one_res.npy'%(TARGET_FOLDER), one_res)
np.save('%s/opencv_res.npy'%(TARGET_FOLDER), opencv_res)
