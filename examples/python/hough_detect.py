#!/usr/bin/env python
#%% Imports
from __future__ import print_function

from PIL import Image
import skimage.color
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage import draw
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))
if len(sys.argv) > 2:
  print('Quiet mode. Plot display disabled.')
  INTERACTIVE = False
else:
  INTERACTIVE = True
import matplotlib.pyplot as plt
import numpy as np
import pickle
import fertilized
import sklearn.metrics
import threading
from hough_preferences import scales, ratios, root_dir, image_dir, annot_dir,\
  test_offset, n_pos, n_threads, min_prob_threshold, n_feature_channels, \
  patch_size, application_step_size, use_reduced_grid
#from joblib import delayed, Parallel

if len(sys.argv) > 1:
  root_dir = sys.argv[1]

SAVE_PREF = 'orig'
VISUALIZATION = INTERACTIVE and n_threads == 1

#%% Load the forest.
with open('forest.pkl', 'r') as df:
  forest = pickle.load(df)
soil = fertilized.Soil()

#%% Read the dataset.
test_ids = list(range(200, 655))

#%% Determine mean box size of training set.
widths = []
heights = []
for idx in list(range(50)) + list(range(100, 150)):
  with open(os.path.join(root_dir, annot_dir, 'bounding-boxes-%03d.txt' % (idx))) as f:
    _ = f.readline()
    l, t, r, b = f.readline().split("\t")
    widths.append(int(r) - int(l))
    heights.append(int(b) - int(t))
box_width = np.mean(widths)
box_height = np.mean(heights)

images = []
im_scores = []
im_truth = []
pos_correct = 0
#%% Test images.
for test_number, test_idx in enumerate(test_ids):
  images.append(np.array(Image.open(os.path.join(root_dir, image_dir, 'image-%03d.png' % (test_idx)))))
  if images[-1].ndim == 2:
    images[-1] = skimage.color.gray2rgb(images[-1])
  else:
    images[-1] = np.ascontiguousarray(images[-1][:, :, :3])

#%%
def evaluate(im_idx, im, lock):
  global pos_correct, im_scores, im_truth
  im_score = -1.
  opt_scale = -1.
  opt_ratio = -1.
  opt_pos = [-1, -1]
  if VISUALIZATION:
    plt.imshow(im)
  for scale in scales:
    scaled_image = np.ascontiguousarray((rescale(im, scale) * 255.).astype('uint8'))
    if VISUALIZATION:
      plt.imshow(scaled_image)
    for ratio in ratios:
      probmap = forest.predict_image(soil.extract_hough_forest_features(scaled_image, (n_feature_channels==32)),
                                     application_step_size,
                                     use_reduced_grid,
                                     ratio,
                                     min_prob_threshold)
      #with open('pmap_%d_%s.pkl' % (im_idx, SAVE_PREF), 'w') as f:
      #  pickle.dump(probmap, f)
      probmap = gaussian(probmap, 3.)
      if VISUALIZATION:
        print("Maximum: ", probmap.max())
        plt.imshow(probmap)
      if probmap.max() > im_score:
        im_score = probmap.max()
        opt_scale = scale
        opt_ratio = ratio
        argmax = np.array(np.unravel_index(np.argmax(probmap), probmap.shape)[:2])
        if not use_reduced_grid:
          opt_pos = argmax / scale
        else:
          opt_pos = (argmax * application_step_size + np.array([patch_size[0]//2, patch_size[1]//2])) / scale
  #############################################################################
  # Evaluation
  with lock:
    im_scores.append(im_score)
    # Intersection over union.
    opt_ratio = opt_ratio ** (1./3.)
    #print opt_ratio
    # x, y coordinates.
    luc = np.array([opt_pos[1] - opt_ratio * 0.5 * box_width,
                    opt_pos[0] - 0.5 * box_height])
    rlc = np.array([opt_pos[1] + opt_ratio * 0.5 * box_width,
                    opt_pos[0] + 0.5 * box_height])
    #print 'luc', luc
    #print 'rlc', rlc
    if im_idx < n_pos:
      with open(os.path.join(root_dir, annot_dir, 'bounding-boxes-%03d.txt' % (im_idx+test_offset))) as f:
        _ = f.readline()
        l, t, r, b = f.readline().split("\t")
      l = int(l)
      t = int(t)
      r = int(r)
      b = int(b)
      #print l, t, r, b
      area_gt = float((r-l) * (b-t))
      #print 'area gt:', area_gt
      assert area_gt > 0.
      intersect_l = max(l, min(r, luc[0]))
      intersect_r = min(r, max(l, rlc[0]))
      assert intersect_r > intersect_l
      intersect_t = max(t, min(b, luc[1]))
      intersect_b = min(b, max(t, rlc[1]))
      #print intersect_l, intersect_t, intersect_r, intersect_b
      area_overlap = float((intersect_r - intersect_l) * (intersect_b - intersect_t))
      assert area_overlap <= area_gt
      area_det = float((rlc[0] - luc[0])*(rlc[1] - luc[1]))
      assert area_overlap <= area_det, '%f, %f' % (area_overlap, area_det)
      #print 'area overlap:', area_overlap
      if area_overlap / (area_det + area_gt - area_overlap) >= 0.5:
        im_truth.append(1)
        pos_correct += 1
        box_pos_correct = True
        print('.', end='')
      else:
        im_truth.append(0)
        im_scores.append(0.)
        im_truth.append(1)
        box_pos_correct = False
        print('X', end='')
    else:
      print ('-', end='')
      im_truth.append(0)
      box_pos_correct = None
  sys.stdout.flush()
  #############################################################################
  # Visualize.
  if VISUALIZATION:
    orig_img = np.array(Image.open(os.path.join(root_dir,
                                                image_dir,
                                                'image-%03d.png' % (im_idx + test_offset))))
    if orig_img.ndim == 2:
      orig_img = skimage.color.gray2rgb(orig_img)
    else:
      orig_img = np.ascontiguousarray(orig_img[:, :, :3])
    # Visualization.
    # point in format (x, y).
    if box_pos_correct is None:
      box_color = (0, 0, 255)
    elif box_pos_correct == True:
      box_color = (0, 255, 0)
    else:
      box_color = (255, 0, 0)
    rr, cc = draw.circle(opt_pos[0], opt_pos[1], 5, shape=orig_img.shape)
    orig_img[rr, cc, :] = (255,255,255)
    luc[1] = max(0, min(luc[1], orig_img.shape[0] - 1))
    luc[0] = max(0, min(luc[0], orig_img.shape[1] - 1))
    rlc[1] = max(0, min(rlc[1], orig_img.shape[0] - 1))
    rlc[0] = max(0, min(rlc[0], orig_img.shape[1] - 1))
    rr, cc = draw.line(int(luc[1]), int(luc[0]), int(luc[1]), int(rlc[0]))
    orig_img[rr, cc, :] = box_color
    rr, cc = draw.line(int(luc[1]), int(rlc[0]), int(rlc[1]), int(rlc[0]))
    orig_img[rr, cc, :] = box_color
    rr, cc = draw.line(int(rlc[1]), int(rlc[0]), int(rlc[1]), int(luc[0]))
    orig_img[rr, cc, :] = box_color
    rr, cc = draw.line(int(rlc[1]), int(luc[0]), int(luc[1]), int(luc[0]))
    orig_img[rr, cc, :] = box_color
    plt.imshow(orig_img)
    plt.show()
  #############################################################################

#%%
lck = threading.Lock()
for im_idx, im in enumerate(images):
  evaluate(im_idx, im, lck)
#Parallel(n_jobs=n_threads, backend='threading')(delayed(evaluate)(im_idx, im, lck) for im_idx, im in enumerate(images))
print('')

#%%
im_scores = np.array(im_scores, dtype='float32')
im_scores /= im_scores.max()
fpr, tpr, thresholds = sklearn.metrics.roc_curve(im_truth, im_scores)

#%%
if VISUALIZATION:
    plt.plot(fpr, tpr)
print('ROC auc score: %f' % (sklearn.metrics.roc_auc_score(im_truth, im_scores)))
print('Correct localizations: %d' % (pos_correct))
print('Ratio: %f' % (float(pos_correct)/float(n_pos)))
assert sklearn.metrics.roc_auc_score(im_truth, im_scores) > 0.68
