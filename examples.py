#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 08:11:04 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import random
import sys

sys.path.append("/home/ryan/github/embs/all_in_one_sentence_embeddings")
from cluster import *
from dimredux import *
from embeddings import *
from prep import *
from readin import *
from viz import *

random.seed(42)

# funcs
prep_load()

# data
dat = fetch_20newsgroups(categories = ["rec.autos"])["data"]

data = []
for d in dat:
	for i in d.split("\n"):
		data.append(i)

# prep
lower_ex = prep_ex(data[:5], prep_lower)
punct_ex = prep_ex(data[:5], prep_punct)
stop_ex = prep_ex(data[:5], prep_stop)
stem_ex = prep_ex(data[:5], prep_stem)
lemma_ex = prep_ex(data[:5], prep_lemma)
spell_ex = prep_ex(data[:5], prep_spell)
clause_ex = prep_ex(data[:5], prep_clause)

# embed
count_ex = pd.DataFrame(
	sk.feature_extraction.text.CountVectorizer(lowercase = False).fit_transform(data[:5]).toarray(),
	columns = sk.feature_extraction.text.CountVectorizer(lowercase = False).fit(data[:5]).get_feature_names_out()
	)
hash_ex = pd.DataFrame(
	sk.feature_extraction.text.HashingVectorizer(lowercase = False, n_features = 2**5).fit_transform(data[:5]).toarray()
	)
tfidf_ex = pd.DataFrame(
	sk.feature_extraction.text.TfidfVectorizer(lowercase = False).fit_transform(data[:5]).toarray(),
	columns = sk.feature_extraction.text.TfidfVectorizer(lowercase = False).fit(data[:5]).get_feature_names_out()
	)
use_ex = pd.DataFrame(model_use(data[:5]))

# cluster
cldata = data[:100]
use_embs = model_use(cldata)

kmeans_ex = cluster_ex(cldata, cluster_kmeans(use_embs, 10, "lloyd"))
affinity_ex = cluster_ex(cldata, cluster_affinity(use_embs))
agglom_ex = cluster_ex(cldata, cluster_agglom(use_embs, 10, "euclidean", "ward"))
birch_ex = cluster_ex(cldata, cluster_birch(use_embs, 50, 10))
dbscan_ex = cluster_ex(cldata, cluster_dbscan(use_embs, 0.5, 5, "euclidean"))
minikmeans_ex = cluster_ex(cldata, cluster_minikmeans(use_embs, 10))
meanshift_ex = cluster_ex(cldata, cluster_meanshift(use_embs, False, False))
optics_ex = cluster_ex(cldata, cluster_optics(use_embs, 5, "minkowski", None))
spectral_ex = cluster_ex(cldata, cluster_spectral(use_embs, 10, "rbf"))
hdbscan_ex = cluster_ex(cldata, cluster_hdbscan(use_embs, 1.0, "euclidean", 5))

# dimredux
tsne_ex = viz_ex(dim_tsne(use_embs, "euclidean", "barnes_hut"), cldata, use_embs)
gaussrandom_ex = viz_ex(dim_gaussrandom(use_embs, 0.1), cldata, use_embs)
sparserandom_ex = viz_ex(dim_sparserandom(use_embs, 0.1), cldata, use_embs)
factor_ex = viz_ex(dim_factor(use_embs, "randomized"), cldata, use_embs)
fastica_ex = viz_ex(dim_fastica(use_embs, "parallel"), cldata, use_embs)
ipca_ex = viz_ex(dim_ipca(use_embs), cldata, use_embs)
kpca_ex = viz_ex(dim_kpca(use_embs, "linear"), cldata, use_embs)
lda_ex = viz_ex(dim_lda(list(map(np.abs, use_embs))), cldata, list(map(np.abs, use_embs)))
minibatchspca_ex = viz_ex(dim_minibatchspca(use_embs, "lars"), cldata, use_embs)
nmf_ex = viz_ex(dim_nmf(list(map(np.abs, use_embs)), None), cldata, list(map(np.abs, use_embs)))
pca_ex = viz_ex(dim_pca(use_embs), cldata, use_embs)
spca_ex = viz_ex(dim_spca(use_embs, "lars"), cldata, use_embs)
tsvd_ex = viz_ex(dim_tsvd(use_embs, "randomized"), cldata, use_embs)
umap_ex = viz_ex(dim_umap(use_embs, 15, 0.1, "euclidean"), cldata, use_embs)



