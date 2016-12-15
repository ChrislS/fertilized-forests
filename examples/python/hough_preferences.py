#%% Parameters
# patch size in format (y, x)
patch_size = (16, 16)
n_trees = 15
min_samples_per_split = 20
max_depth = 15
n_splits = 2000
n_thresholds = 10
scales = (0.7,) #(0.7, 0.6, 0.5, 0.4, 0.3)
ratios = (1.,) #(0.5, 0.75, 1., 1.25, 1.5)
n_samples_pos = 2
n_samples_neg = 2
n_feature_channels = 15
max_patches_per_node = 40000

#################
n_threads = 1
#################

min_prob_threshold = 0.5
test_offset = 200
application_step_size = 2
use_reduced_grid = True
n_pos = 228
# You can find the dataset on
# http://jamie.shotton.org/work/data/WeizmannMultiScale.zip.
# Simply unzip it in the following configured folder.
root_dir = r'WeizmannHorse'
image_dir = 'Images'
annot_dir = 'BoundingBoxes'
