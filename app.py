#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 08:33:29 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import random
import streamlit as st

from cluster import *
from dimredux import *
from embeddings import *
from eval import *
from prep import *
from readin import *
from viz import *

st.set_page_config(layout="wide")
random.seed(42)

st.title("All* in One Sentence Embeddings")
with st.expander("Notes", False):
	st.markdown("""
This tool brings together a wide range of preprocessing, embedding, clustering, and dimensionality reduction techniques into one place, to make comparison of approaches quick and easy. The hope is that this will be useful to people working with natural language data in a range of fields, by removing a significant amount of work in the early stages of processing a dataset. 

**A couple notes on the following sections and their design/usage:**

- This tool is aimed primarily at users who have real-world data and real-world problems to solve. Many of the below notes stem from this choice, and while it may serve as a useful jumping off point for demonstrating differences in clustering/embedding/dimensionality reduction techniques, the *theory* behind those techniques is secondary here.
- That being said, this app is not a substitute for a working understanding of the techniques used. Not every clustering algorithm is appropriate for every data type or every task, and explaining the right approach for the task in front of you is way beyond the scope of this project.
- The visualization included at the bottom of the page will be most useful for higher dimensional embedding models such as the SentenceTransformers and USE options. In all cases, however, the primary evaluation of embedding and clustering effectiveness should be done via the "Clustered Data" table at the top of the page.
- This tool does not encompass all possible arguments across all of the individual functions in the pipeline. Fine tuning of model parameters should be done in a separate environment, but as much as possible the parameters which should be the most impactful are included. 
- Because of the range of possible metrics and algorithms here, it is expected that some of the many, many possible combinations will be incompatible.

Additionally, this app was created in parallel with a static report published via Datapane, available [here](). The report goes into some detail on the conceptual underpinnings and usage of the techniques discussed here.
""")

#sidebar
with st.sidebar:
	st.title("Pipeline Configuration")

##data import
	with st.expander("Data Selection", True):
		source = st.radio("Source", ("Default","File Upload","Paste Text","Scikit-Learn Dataset"))
		head = st.selectbox("Head", ("All", 50, 100, 500, 1000), help = "Use first n rows of selected data")

		if source == "File Upload":
			upl = st.file_uploader("Upload a CSV")
			if upl:
				text = list(pd.read_csv(upl))
			else :
				text = default
		if source == "Paste Text":
			delim = st.text_input("Delimiter", ",", help = "Delimiting character for pasted text")
			inp = st.text_area("Paste Text Here", "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair")
			text = inp.split(delim)
		if source == "Scikit-Learn Dataset":
			cat = st.multiselect("Dataset Selection", ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc'], default = "alt.atheism", help = "Text datasets provided via sklearn.datasets.fetch_20newsgroups()")
			text = fetch_20newsgroups(categories = cat)["data"]
		if source == "Default":
			text = default
		if head != "All":
			text = text[:head]

## preprocessing
	with st.expander("Text Preprocessing", False):
		prep = text
		pre = st.multiselect("Preprocessing Steps", ["Lowercase","Punctuation","Stopwords","Lemmatization","Stemming","Spelling","Clause Separation"], help = "Preprocessing steps to apply to provided data")
		prep_load()

		if "Lowercase" in pre:
			prep = prep_lower(prep)
		if "Punctuation" in pre:
			prep = prep_punct(prep)
		if "Stopwords" in pre:
			prep = prep_stop(prep)
		if "Lemmatization" in pre:
			prep = prep_lemma(prep)
		if "Stemming" in pre:
			prep = prep_stem(prep)
		if "Spelling" in pre:
			prep = prep_spell(prep)
		if "Clause Separation" in pre:
			clause_reg_box = st.text_input("clause sep regex", clause_reg, help = "Regex defining separation of clauses within each sentence/line")
			clause_word_box = st.text_input("clause sep words", clause_words, help = "Words indicating a clause boundary")
			clause_sep = f"{clause_reg}{' | '.join(clause_words)}".replace("] ", "]")
			prep = prep_clause(prep)

## embeddings
	with st.expander("Sentence Embeddings", False):
		mod = st.selectbox("Embedding Algorithm", ("tf-idf","Hash","Count","SentenceTransformers Model","Universal Sentence Encoder"), help = "Algorithm used for sentence embeddings, preprocessing steps may be duplicated between the abova and the following models")

		if mod == "tf-idf":
			ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
			emb = model_tfidf(prep, (1,ng))
		if mod == "Hash":
			ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
			emb = model_hash(prep, (1,ng))
		if mod == "Count":
			ng = st.slider("ngram_range", 1, 5, help = "Break sentences into chunks ranging in length from 1 to n. This may add some contextual information in the embeddings for bag-of-words based algorithms")
			emb = model_count(prep, (1,ng))
		if mod == "SentenceTransformers Model":
			st_mod = st.selectbox("st model selection", st_available_models, help = "Pretrained models available through the SetnenceTransformers library and HuggingFace.co")
			emb = model_snt(prep, st_mod)
		if mod == "Universal Sentence Encoder":
			emb = model_use(prep)

## clustering
	with st.expander("Sentence Clustering", False):
		clu = st.selectbox("Clustering Algorithm", ("Affinity Propagation","Agglomerative Clustering","Birch","DBSCAN","HDBSCAN","KMeans","Mini Batch KMeans","Mean Shift","OPTICS","Spectral Clustering"), help = "Algorithm to use to group similar datapoints together")

		if clu == "Affinity Propagation":
			cl = cluster_affinity(emb)
		if clu == "Agglomerative Clustering":
			aff = st.radio("affinity", ("euclidean", "l1", "l2", "manhattan", "cosine"), help = "Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. If linkage is “ward”, only “euclidean” is accepted")
			ncl = st.slider("n_clusters", 1, 20, 10, help = "The number of clusters to find")
			lnk = st.radio("linkage", ("ward", "complete", "average", "single"), help = "Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. ‘ward’ minimizes the variance of the clusters being merged. ‘average’ uses the average of the distances of each observation of the two sets. ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets. ‘single’ uses the minimum of the distances between all observations of the two sets.")
			cl = cluster_agglom(emb, ncl, aff, lnk)
		if clu == "Birch":
			bf = st.slider("branching factor", 0, 100, 50, help = "Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes with the subclusters redistributed in each. The parent subcluster of that node is removed and two new subclusters are added as parents of the 2 split nodes.")
			ncl = st.slider("n_clusters", 1, 20, 10, help = "The number of clusters to find")
			cl = cluster_birch(emb, bf, ncl)
		if clu == "DBSCAN":
			mtrc = st.selectbox("metric", metrics_list, help = "The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is “precomputed”, X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors for DBSCAN.")
			eps_cl = st.slider("eps", 0.0, 2.0, step = .001, value = 1.0, help = "The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.")
			mins = st.slider("min samples", 1, 20, 5, help = "The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.")
			cl = cluster_dbscan(emb, eps_cl, mins, mtrc)
		if clu == "HDBSCAN":
			alp = st.slider("alpha", 0.0, 2.0, step = .001, value = 1.0, help = "A distance scaling parameter as used in robust single linkage")
			mtrc = st.selectbox("metric", metrics_list, help = "The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by metrics.pairwise.pairwise_distances for its metric parameter.")
			mincl = st.slider("min cluster size", 1, 20, 5, help = "The minimum size of clusters; single linkage splits that contain fewer points than this will be considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.")
			cl = cluster_hdbscan(emb, alp, mtrc, mincl)
		if clu == "KMeans":
			alg = st.radio("affinity", ("elkan", "lloyd"), help = "K-means algorithm to use. The classical EM-style algorithm is 'lloyd'. The 'elkan' variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).")
			ncl = st.slider("n_clusters", 1, 20, 10, help = "The number of clusters to find")
			cl = cluster_kmeans(emb, ncl, alg)
		if clu == "Mini Batch KMeans":
			ncl = st.slider("n_clusters", 1, 20, 10, help = "The number of clusters to find")
			cl = cluster_minikmeans(emb, ncl)
		if clu == "Mean Shift":
			sdng = st.radio("bin_seeding", (True, False), help = "If true, initial kernel locations are not locations of all points, but rather the location of the discretized version of points, where points are binned onto a grid whose coarseness corresponds to the bandwidth. Setting this option to True will speed up the algorithm because fewer seeds will be initialized.")
			cl_all = st.radio("cluster_all", (True, False), help = "If true, then all points are clustered, even those orphans that are not within any kernel. Orphans are assigned to the nearest kernel. If false, then orphans are given cluster label -1.")
			cl = cluster_meanshift(emb, sdng, cl_all)
		if clu == "OPTICS":
			mtrc = st.selectbox("metric", metrics_list, help = "Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used. If metric is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two arrays as input and return one value indicating the distance between them. This works for Scipy’s metrics, but is less efficient than passing the metric name as a string.")
			mins = st.slider("min samples", 1, 20, 5, help = "The number of samples in a neighborhood for a point to be considered as a core point. Also, up and down steep regions can’t have more than min_samples consecutive non-steep points. Expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).")
			mincl = st.slider("min cluster size", 1, 20, help = "Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2). If None, the value of min_samples is used instead. Used only when cluster_method='xi'.")
			cl = cluster_optics(emb, mins, mtrc, mincl)
		if clu == "Spectral Clustering":
			aff = st.radio("affinity", ("nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors", "additive_chi2", "chi2", "linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"), help = "How to construct the affinity matrix. ‘nearest_neighbors’: construct the affinity matrix by computing a graph of nearest neighbors. ‘rbf’: construct the affinity matrix using a radial basis function (RBF) kernel. ‘precomputed’: interpret X as a precomputed affinity matrix, where larger values indicate greater similarity between instances. ‘precomputed_nearest_neighbors’: interpret X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance. One of the kernels supported by pairwise_kernels.")
			ncl = st.slider("n_clusters", 1, 20, 10, help = "The number of clusters to find")
			cl = cluster_spectral(emb, ncl, aff)

## dimredux
	with st.expander("Dimensionality Reduction", False):
		dim = st.selectbox("Algorithm", ("t-SNE","Gaussian Random Projection","Sparse Random Projection","Factor Analysis","Fast ICA","Incremental PCA","Kernel PCA","Latent Dirichlet Allocation","Mini Batch Sparse PCA","NMF","PCA","Sparse PCA","Truncated SVD","UMAP"), help = "Algorithm to use to reduce embeddings in high dimensional vector space to 2 or 3 dimensions, useful for visualization")

		if dim == "t-SNE":
			mtrc = st.radio("metric", ("cityblock","cosine","euclidean","haversine","l1","l2","manhattan","nan_euclidean"), help = "The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is assumed to be a distance matrix. Alternatively, if metric is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two arrays from X as input and return a value indicating the distance between them. The default is “euclidean” which is interpreted as squared euclidean distance.")
			mth = st.radio("method", ("barnes_hut", "exact"), help = "By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale to millions of examples.")
			d2,d3 = dim_tsne(emb, mtrc, mth)
		if dim == "Gaussian Random Projection":
			eps_dim = st.slider("eps", 0.0, 2.0, step = .001, value = 1.0, help = "Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma when n_components is set to ‘auto’. The value should be strictly positive. Smaller values lead to better embedding and higher number of dimensions (n_components) in the target projection space.")
			d2,d3 = dim_gaussrandom(emb, eps_dim)
		if dim == "Sparse Random Projection":
			eps_dim = st.slider("eps", 0.0, 2.0, step = .001, value = 1.0, help = "Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma when n_components is set to ‘auto’. This value should be strictly positive. Smaller values lead to better embedding and higher number of dimensions (n_components) in the target projection space.")
			d2,d3 = dim_sparserandom(emb, eps_dim)
		if dim == "Factor Analysis":
			mth = st.radio("svd_method", ("lapack", "randomized"), help = "Which SVD method to use. If ‘lapack’ use standard SVD from scipy.linalg, if ‘randomized’ use fast randomized_svd function. Defaults to ‘randomized’. For most applications ‘randomized’ will be sufficiently precise while providing significant speed gains. Accuracy can also be improved by setting higher values for iterated_power. If this is not sufficient, for maximum precision you should choose ‘lapack’.")
			d2,d3 = dim_factor(emb, mth)
		if dim == "Fast ICA":
			alg = st.radio("algorithm", ("parallel", "deflation"), help = "Apply parallel or deflational algorithm for FastICA.")
			d2,d3 = dim_fastica(emb, alg)
		if dim == "Incremental PCA":
			d2,d3 = dim_ipca(emb)
		if dim == "Kernel PCA":
			krnl = st.radio("kernel", ("linear","poly","rbf","sigmoid","cosine"), help = "Kernel used for PCA.")
			d2,d3 = dim_kpca(emb, krnl)
		if dim == "Latent Dirichlet Allocation":
			d2,d3 = dim_lda(emb)
		if dim == "Mini Batch Sparse PCA":
			mth = st.radio("method", ("lars", "cd"), help = "Method to be used for optimization. lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso). Lars will be faster if the estimated components are sparse.")
			d2,d3 = dim_minibatchspca(emb, mth)
		if dim == "NMF":
			nmf_init = st.radio("init", ("random", "nndsvd", "nndsvda", "nndsvdar"), help = "Method used to initialize the procedure. Default: None. Valid options: None: ‘nndsvda’ if n_components <= min(n_samples, n_features), otherwise random. 'random': non-negative random matrices, scaled with: sqrt(X.mean() / n_components). 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD) initialization (better for sparseness). 'nndsvda': NNDSVD with zeros filled with the average of X (better when sparsity is not desired). 'nndsvdar' NNDSVD with zeros filled with small random values (generally faster, less accurate alternative to NNDSVDa for when sparsity is not desired). 'custom': use custom matrices W and H")
			d2,d3 = dim_nmf(emb, nmf_init)
		if dim == "PCA":
			d2,d3 = dim_pca(emb)
		if dim == "Sparse PCA":
			mth = st.radio("method", ("lars", "cd"), help = "Method to be used for optimization. lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso). Lars will be faster if the estimated components are sparse.")
			d2,d3 = dim_spca(emb, mth)
		if dim == "Truncated SVD":
			alg = st.radio("algorithm", ("arpack","randomized"), help = "SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds), or “randomized” for the randomized algorithm due to Halko (2009).")
			d2,d3 = dim_tsvd(emb, alg)
		if dim == "UMAP":
			mind = st.slider("min distance", 0.0, 2.0, 0.1, help = "The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points. The value should be set relative to the spread value, which determines the scale at which embedded points will be spread out.")
			nne = st.slider("n_neighbors", 2, 200, 10, help = "The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100.")
			mtrc = st.radio("metric", ("euclidean","manhattan","chebyshev","minkowski","canberra","braycurtis","mahalanobis","wminkowski","seuclidean","cosine","correlation","haversine","hamming","jaccard","dice","russelrao","kulsinski","ll_dirichlet","hellinger","rogerstanimoto","sokalmichener","sokalsneath","yule"), help = "The metric to use to compute distances in high dimensional space. If a string is passed it must match a valid predefined metric. If a general metric is required a function that takes two 1d arrays and returns a float can be provided. For performance purposes it is required that this be a numba jit’d function. ")
			d2,d3 = dim_umap(emb, nne, mind, mtrc)

emb_df = pd.DataFrame({
	"prep" : prep, 
	"cluster" : cl, 
	"d2" : list(d2),
	"d3" : list(d3), 
	"emb" : list(emb)
	})
emb_display = emb_df[["prep","cluster"]].groupby("cluster").agg(lambda x: list(x)).reset_index()
emb_display["n"] = [len(i) for i in emb_display["prep"]]

#body
with st.container():
	c1, c2 = st.columns(2)
	with c1:
		##raw data
		st.subheader("Raw and Preprocessed Data")
		st.write(dict(zip(text[:5],prep[:5])))
	with c2:
		##embedings
		st.subheader("Clustered Data")
		st.dataframe(emb_display[["cluster","n","prep"]])

with st.container():
	st.header("Clustering Metrics & Plots")

	c3, c4, c5 = st.columns(3)
	with c3:
		st.metric("Calinski Harabasz Score", eval_ch(emb, cl))
		st.caption("A higher Calinski-Harabasz score relates to a model with better defined clusters. The index is the ratio of the sum of between-clusters dispersion and of within-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared)")
	with c4:
		st.metric("Davies Bouldin Score", eval_db(emb, cl))
		st.caption("A lower Davies-Bouldin index relates to a model with better separation between the clusters. This index signifies the average ‘similarity’ between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves. Zero is the lowest possible score. Values closer to zero indicate a better partition.")
	with c5:
		st.metric("Silhouette Score", eval_s(emb, cl))
		st.caption("A higher Silhouette Coefficient score relates to a model with better defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores: a: The mean distance between a sample and all other points in the same class. b: The mean distance between a sample and all other points in the next nearest cluster.")

	c6, c7 = st.columns(2)
	with c6:
		st.plotly_chart(d2_plot(emb_df), use_container_width = False, help = "Embeddings Reduced to 2 Dimensions")
	with c7:
		st.plotly_chart(d3_plot(emb_df), use_container_width = False, help = "Embeddings Reduced to 3 Dimensions")
