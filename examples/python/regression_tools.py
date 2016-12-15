# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:10:22 2014

@author: einfalmo
"""
import numpy as np
import numpy.matlib as npmat

def likelihood_from_norm_dist (test_vec, mean_vec, covar_mat):
  if ((test_vec.shape != mean_vec.shape) or \
      (mean_vec.shape[0] != covar_mat.shape[0]) or \
      (covar_mat.shape[0] != covar_mat.shape[1])):
      raise Exception ("Invalid dimensions.")
  dist_dim = mean_vec.shape[0]
  diff_vec = test_vec - mean_vec    
  likelihood = (1/(np.sqrt(pow(2*np.pi,dist_dim) * covar_mat.diagonal().prod()))) * \
                  np.exp(-0.5 * diff_vec.T * covar_mat.I * diff_vec)
  
  return likelihood.item(0,0)

def prob_dist_from_tree_result(tree_result, sample_index):
  if (not isinstance(tree_result, np.ndarray)):
      raise Exception("Invalid tree_result.")
  n_samples=tree_result.shape[0]
  if (sample_index >= n_samples):
      raise Exception("Invalid sample index.")
  annot_dim = tree_result.shape[2]
  
  mean = np.mat(tree_result[sample_index,0,:], dtype=tree_result.dtype)
  if   (mean.shape[1] > 1):
      mean = mean.T
  covar_mat = npmat.zeros((annot_dim,annot_dim), dtype=tree_result.dtype)
  mat_diag_entries = tree_result[sample_index,1,:]
  for i in range(annot_dim):
      covar_mat[i,i] = mat_diag_entries[i]
  
  return mean, covar_mat
    
def likelihood_from_tree_result(tree_result, sample_index, test_vec=None):
  if (not isinstance(tree_result, np.ndarray)):
      raise Exception("Invalid tree_result.")
  n_samples = tree_result.shape[0]
  if (sample_index >= n_samples):
      raise Exception("Invalid sample index.")
  mean_vec, covar_mat = prob_dist_from_tree_result(tree_result, sample_index)
  likelihood = 0
  if (test_vec != None):
    likelihood = likelihood_from_norm_dist(test_vec, mean_vec, covar_mat)
  else:
    likelihood = likelihood_from_norm_dist(mean_vec, mean_vec, covar_mat)
  
  return likelihood
    
    
def likelihood_from_forest_result(forest_result ,sample_index, test_vec=None):
  if (not isinstance(forest_result,np.ndarray)):
      raise Exception("Invalid tree_result.")
  n_samples = forest_result.shape[0]
  if (sample_index >= n_samples):
      raise Exception("Invalid sample index.")
  n_trees = forest_result.shape[1]
  likelihood = 0
  for t in range(n_trees):
      mean_vec, covar_mat = prob_dist_from_tree_result(forest_result[:,t,0:2,:], sample_index)
      if (test_vec != None):
          likelihood += likelihood_from_norm_dist(test_vec, mean_vec, covar_mat) / n_trees
      else:
          likelihood += likelihood_from_norm_dist(mean_vec, mean_vec, covar_mat) / n_trees
      
  
  return likelihood
  

def mean_from_forest_result(forest_result):
  return np.mean(forest_result[:,:,0,:], axis=1, dtype='float64')
  

  
