#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:46:02 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import random

from hdbscan import HDBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

random.seed(42)

def cluster_kmeans(in_embs, n_clusters, algorithm):
	"""
KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
	"""
	return KMeans(n_clusters = n_clusters, algorithm = algorithm).fit(in_embs).labels_

def cluster_affinity(in_embs):
	"""
AffinityPropagation(*, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state=None)[source]¶
	"""
	return AffinityPropagation().fit(in_embs).labels_

def cluster_agglom(in_embs, n_clusters, affinity, linkage):
	"""
AgglomerativeClustering(n_clusters=2, *, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)[source]¶
	"""
	return AgglomerativeClustering(n_clusters = n_clusters, affinity = affinity, linkage = linkage).fit(in_embs).labels_

def cluster_birch(in_embs, branching_factor, n_clusters):
	"""
Birch(*, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)[source]¶
	"""
	return Birch(branching_factor = branching_factor, n_clusters = n_clusters).fit(in_embs).labels_

def cluster_dbscan(in_embs, eps, min_samples, metric):
	"""
DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)[source]¶
	"""
	return DBSCAN(eps = eps, min_samples = min_samples, metric = metric).fit(in_embs).labels_

def cluster_minikmeans(in_embs, n_clusters):
	"""
MiniBatchKMeans(n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)[source]¶
	"""
	return MiniBatchKMeans(n_clusters = n_clusters).fit(in_embs).labels_

def cluster_meanshift(in_embs, bin_seeding, cluster_all):
	"""
MeanShift(*, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
	"""
	return MeanShift(bin_seeding = bin_seeding, cluster_all = cluster_all).fit(in_embs).labels_

def cluster_optics(in_embs, min_samples, metric, min_cluster_size):
	"""
OPTICS(*, min_samples=5, max_eps=inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None, n_jobs=None)[source]¶
	"""
	return OPTICS(min_samples = min_samples, metric = metric, min_cluster_size = min_cluster_size).fit(in_embs).labels_

def cluster_spectral(in_embs, n_clusters, affinity):
	"""
SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)[source]¶
	"""
	return SpectralClustering(n_clusters = n_clusters, affinity = affinity).fit(in_embs).labels_

def cluster_hdbscan(in_embs, alpha, metric, min_cluster_size):
	"""
HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True, gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None), metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
	"""
	return HDBSCAN(alpha = alpha, metric = metric, min_cluster_size = min_cluster_size).fit(in_embs).labels_

metrics_list = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]

def cluster_ex(in_text, labels):
	out = pd.DataFrame({"Text" : in_text , "Cluster" : labels}).sort_values(by = "Cluster")
	return out