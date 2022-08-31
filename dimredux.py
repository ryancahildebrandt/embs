#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:46:20 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import random
import sklearn as sk
import sklearn.manifold
import umap

from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

random.seed(42)

def dim_tsne(in_embs, metric, method):
	"""
TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='warn', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='deprecated')[source]¶
	"""
	d2 = sk.manifold.TSNE(n_components = 2, metric = metric, method = method).fit_transform(in_embs)
	d3 = sk.manifold.TSNE(n_components = 3, metric = metric, method = method).fit_transform(in_embs)
	return [d2,d3]

def dim_gaussrandom(in_embs, eps):
	"""
GaussianRandomProjection(n_components='auto', *, eps=0.1, compute_inverse_components=False, random_state=None)[source]¶
	"""
	d2 = GaussianRandomProjection(n_components = 2, eps = eps).fit_transform(in_embs)
	d3 = GaussianRandomProjection(n_components = 3, eps = eps).fit_transform(in_embs)
	return [d2,d3]

def dim_sparserandom(in_embs, eps):
	"""
SparseRandomProjection(n_components='auto', *, density='auto', eps=0.1, dense_output=False, compute_inverse_components=False, random_state=None)[source]¶
	"""
	d2 = SparseRandomProjection(n_components = 2, eps = eps).fit_transform(in_embs)
	d3 = SparseRandomProjection(n_components = 3, eps = eps).fit_transform(in_embs)
	return [d2,d3]

def dim_factor(in_embs, svd_method):
	"""
FactorAnalysis(n_components=None, *, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, rotation=None, random_state=0)[source]¶
	"""
	d2 = FactorAnalysis(n_components = 2, svd_method = svd_method).fit_transform(in_embs)
	d3 = FactorAnalysis(n_components = 3, svd_method = svd_method).fit_transform(in_embs)
	return [d2,d3]

def dim_fastica(in_embs, algorithm):
	"""
FastICA(n_components=None, *, algorithm='parallel', whiten='warn', fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)[source]¶
	"""
	d2 = FastICA(n_components = 2, algorithm = algorithm).fit_transform(in_embs)
	d3 = FastICA(n_components = 3, algorithm = algorithm).fit_transform(in_embs)
	return [d2,d3]

def dim_ipca(in_embs):
	"""
IncrementalPCA(n_components=None, *, whiten=False, copy=True, batch_size=None)[source]¶
	"""
	d2 = IncrementalPCA(n_components = 2).fit_transform(in_embs)
	d3 = IncrementalPCA(n_components = 3).fit_transform(in_embs)
	return [d2,d3]

def dim_kpca(in_embs, kernel):
	"""
KernelPCA(n_components=None, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, iterated_power='auto', remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)[source]¶
	"""
	d2 = KernelPCA(n_components = 2, kernel = kernel).fit_transform(in_embs)
	d3 = KernelPCA(n_components = 3, kernel = kernel).fit_transform(in_embs)
	return [d2,d3]

def dim_lda(in_embs):
	"""
LatentDirichletAllocation(n_components=10, *, doc_topic_prior=None, topic_word_prior=None, learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=- 1, total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None)[source]¶
	"""
	d2 = LatentDirichletAllocation(n_components = 2).fit_transform(in_embs)
	d3 = LatentDirichletAllocation(n_components = 3).fit_transform(in_embs)
	return [d2,d3]

def dim_minibatchspca(in_embs, method):
	"""
MiniBatchSparsePCA(n_components=None, *, alpha=1, ridge_alpha=0.01, n_iter=100, callback=None, batch_size=3, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None)[source]¶
	"""
	d2 = MiniBatchSparsePCA(n_components = 2, method = method).fit_transform(in_embs)
	d3 = MiniBatchSparsePCA(n_components = 3, method = method).fit_transform(in_embs)
	return [d2,d3]

def dim_nmf(in_embs, init):
	"""
NMF(n_components=None, *, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha='deprecated', alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0, shuffle=False, regularization='deprecated')[source]¶
	"""
	d2 = NMF(n_components = 2, init = init).fit_transform(in_embs)
	d3 = NMF(n_components = 3, init = init).fit_transform(in_embs)
	return [d2,d3]

def dim_pca(in_embs):
	"""
PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)[source]¶
	"""
	d2 = PCA(n_components = 2).fit_transform(in_embs)
	d3 = PCA(n_components = 3).fit_transform(in_embs)
	return [d2,d3]

def dim_spca(in_embs, method):
	"""
SparsePCA(n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None)[source]¶
	"""
	d2 = SparsePCA(n_components = 2, method = method).fit_transform(in_embs)
	d3 = SparsePCA(n_components = 3, method = method).fit_transform(in_embs)
	return [d2,d3]

def dim_tsvd(in_embs, algorithm):
	"""
TruncatedSVD(n_components=2, *, algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', random_state=None, tol=0.0)[source]¶
	"""
	d2 = TruncatedSVD(n_components = 2, algorithm = algorithm).fit_transform(in_embs)
	d3 = TruncatedSVD(n_components = 3, algorithm = algorithm).fit_transform(in_embs)
	return [d2,d3]

def dim_umap(in_embs, n_neighbors, min_dist, metric):
	"""
UMAP(n_neighbors=15, n_components=2, metric='euclidean', metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True, n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None))
	"""
	d2 = umap.UMAP(n_components = 2, n_neighbors = n_neighbors, min_dist = min_dist, metric = metric).fit_transform(in_embs)
	d3 = umap.UMAP(n_components = 3, n_neighbors = n_neighbors, min_dist = min_dist, metric = metric).fit_transform(in_embs)
	return [d2,d3]