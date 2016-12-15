#!/usr/bin/env python
#%% Imports
from PIL import Image
import skimage.color
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
import fertilized
import pickle
from hough_preferences import patch_size, n_trees, min_samples_per_split, \
  max_depth, n_splits, n_thresholds, n_samples_pos, n_samples_neg, \
  n_feature_channels, max_patches_per_node, n_threads, root_dir, image_dir, \
  annot_dir

if len(sys.argv) > 1:
  root_dir = sys.argv[1]

np.random.seed(8)

#%% Read the dataset.
train_pos_ids = list(range(50)) + list(range(100, 150))
train_neg_ids = list(range(50, 100)) + list(range(150, 200))
annotations = np.zeros((n_samples_pos * len(train_pos_ids) +\
                        n_samples_neg * len(train_neg_ids), 5), dtype='int16')
images = []
#%% Positive images.
for pos_number, pos_idx in enumerate(train_pos_ids):
  images.append(np.array(Image.open(os.path.join(root_dir, image_dir, 'image-%03d.png' % (pos_idx)))))
  assert not images[-1] is None
  # Read the annotation file.
  with open(os.path.join(root_dir, annot_dir, 'bounding-boxes-%03d.txt' % (pos_idx))) as f:
    _ = f.readline()
    l, t, r, b = f.readline().split("\t")
  l = int(l)
  t = int(t)
  r = int(r)
  b = int(b)
  assert t + patch_size[0] // 2 + 1 < b - patch_size[0] // 2
  annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 0] = pos_number
  annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 1] = \
                np.random.randint(l + patch_size[1] // 2,
                                  r - patch_size[1] // 2,
                                  size=(n_samples_pos,))
  assert np.all(annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 1] - l >= patch_size[1] // 2)
  assert np.all(r - annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 1] > patch_size[1] // 2)
  annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 2] = \
                np.random.randint(t + patch_size[0] // 2,
                                  b - patch_size[0] // 2,
                                  size=(n_samples_pos,))
  assert np.all(annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 2] - t >= patch_size[0] // 2)
  assert np.all(b - annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 2] > patch_size[0] // 2)
  annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 3] = \
                int(float(l+r)/2.) - annotations[pos_number*n_samples_pos : (pos_number+1)*n_samples_pos, 1]
  annotations[pos_number * n_samples_pos : (pos_number+1)*n_samples_pos, 4] = \
                int(float(t+b)/2.) - annotations[pos_number*n_samples_pos : (pos_number+1)*n_samples_pos, 2]

pos_offset = n_samples_pos * len(train_pos_ids)
#%% Negative images.
for neg_number, neg_idx in enumerate(train_neg_ids):
  images.append(np.array(Image.open(os.path.join(root_dir, image_dir, 'image-%03d.png' % (neg_idx)))))
  assert not images[-1] is None
  annotations[pos_offset+neg_number*n_samples_neg : pos_offset+(neg_number+1)*n_samples_neg, 0] = neg_number + len(train_pos_ids)
  annotations[pos_offset+neg_number*n_samples_neg : pos_offset+(neg_number+1)*n_samples_neg, 1] = \
                np.random.randint(patch_size[1] // 2,
                                  images[-1].shape[1] - patch_size[1] // 2,
                                  size=(n_samples_neg,))
  annotations[pos_offset+neg_number*n_samples_neg : pos_offset+(neg_number+1)*n_samples_neg, 2] = \
                np.random.randint(patch_size[0] // 2,
                                  images[-1].shape[0] - patch_size[0] // 2,
                                  size=(n_samples_neg,))
#%% Feature extraction.
soil = fertilized.Soil('uint8', 'int16', 'int16', fertilized.Result_Types.hough_map)
if os.path.isfile('feat_images.pkl'):
    with open('feat_images.pkl', 'r') as f:
        feat_images = pickle.load(f)
else:
    cvimages = []
    for im in images:
      if im.ndim == 2:
        cvimages.append(np.ascontiguousarray(skimage.color.gray2rgb(im)))
      else:
        cvimages.append(np.ascontiguousarray(im[:, :, :3]))
    feat_images = [soil.extract_hough_forest_features(im, full=(n_feature_channels==32)) for im in cvimages]
    with open('feat_images.pkl', 'w') as f:
        pickle.dump(feat_images, f)

#%% Dataset visualization.
#for i in xrange(len(images)):
#  plt.imshow(images[i])

#%% Forest construction.
random_init = 1
trees = []
for tree_idx in range(n_trees):
  print('Constructing and training tree %d.' % (tree_idx))
  random_seed = tree_idx * 2 + 1 + random_init * n_trees
  sman = soil.NoCopyPatchSampleManager(
            feat_images,
            annotations,
            n_samples_pos * len(train_pos_ids),
            n_feature_channels,
            patch_size[0],
            patch_size[1],
            False)
  dprov = soil.SubsamplingDataProvider(max_patches_per_node,
                                       sman,
                                       random_seed)
  tree = soil.StandardHoughTree((patch_size[1],patch_size[0],n_feature_channels),
                                n_thresholds,
                                n_splits,
                                max_depth,
                                (1, min_samples_per_split),
                                random_seed,
                                (0.,0.),
                                patch_annot_luc=False,
                                allow_redraw=False,
                                num_threads=n_threads,
                                entropy_names=['shannon', 'shannon'])
  tree.fit_dprov(dprov, True)
  trees.append(tree)

#%%
forest = soil.CombineTrees(trees)
with open('forest.pkl', 'w') as df:
  pickle.dump(forest, df)
