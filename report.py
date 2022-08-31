#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 08:33:35 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import datapane as dp
import markdown as md
import random
import re

from examples import *

random.seed(42)

#md readin
with open('report.md', 'r') as f:
    report_md = f.read()

#funcs
def pull_section(section_title):
	out = re.search(f"(#* )({section_title}\n)(.*?)#", report_md, re.DOTALL).group(3)
	return out

def content_page(section_title, blocks):
	out = dp.Page(
		title = section_title,
		blocks = [
		dp.Text(f"# {section_title}"),
		dp.Select(
			blocks = [dp.Text(pull_section(section_title), label = "Overview")] + blocks,
			type = dp.SelectType.TABS)
		])
	return out

def content_group(section_title, ex):
	out = dp.Group(
		dp.Text(pull_section(section_title)),
		ex,
		label = section_title)
	return out

def dim_plots(ex):
	out = dp.Group(
		ex["d2"],
		ex["d3"],
		columns = 2
		)
	return out

#report gen
rprt = dp.Report(
	dp.Page(
		title = "Overview",
		blocks = [
		dp.Text("# The Ex-Academic's Sentence Embedding Guide"),
		dp.Select(
			blocks = [
			dp.Text(pull_section("Overview"), label = "Overview"),
			dp.Text(pull_section("Getting Started"), label = "Getting Started"),
			dp.Text(pull_section("Sources"), label = "Sources")
			], 
			type = dp.SelectType.TABS)
		]),
	content_page("Preprocessing", [
		content_group("Lowercasing", dp.Table(lower_ex)),
		content_group("Punctuation Removal", dp.Table(punct_ex)),
		content_group("Stopword Removal", dp.Table(stop_ex)),
		content_group("Stemming", dp.Table(stem_ex)),
		content_group("Lemmatization", dp.Table(lemma_ex)),
		content_group("Spelling Correction", dp.Table(spell_ex)),
		content_group("Clause Separation", dp.Table(clause_ex))
		]),
	content_page("Embedding", [
		content_group("Count", dp.Table(count_ex)),
		content_group("Hash", dp.Table(hash_ex)),
		content_group("TF-IDF", dp.Table(tfidf_ex)),
		content_group("Sentence Transformers & Universal Sentence Encoder", dp.Table(use_ex))
		]),
	content_page("Clustering", [
		content_group("Affinity Propagation", dp.DataTable(affinity_ex)),
		content_group("Agglomerative", dp.DataTable(agglom_ex)),
		content_group("Birch", dp.DataTable(birch_ex)),
		content_group("DBSCAN & HDBSCAN", dp.Group(
			dp.Text("### DBSCAN"), dp.DataTable(dbscan_ex), 
			dp.Text("### HDBSCAN"), dp.DataTable(hdbscan_ex))),
		content_group("KMeans & Mini Batch KMeans", dp.Group(
			dp.Text("### KMeans"), dp.DataTable(kmeans_ex), 
			dp.Text("### Mini Batch KMeans"), dp.DataTable(minikmeans_ex))),
		content_group("Mean Shift", dp.DataTable(meanshift_ex)),
		content_group("OPTICS", dp.DataTable(optics_ex)),
		content_group("Spectral", dp.DataTable(spectral_ex))
		]),
	content_page("Dimensionality Reduction", [
		content_group("Uniform Manifold Approximation & Projection", dim_plots(umap_ex)),
		content_group("Truncated Singular Value Decomposition", dim_plots(tsvd_ex)),
		content_group("Principal Component Analysis & Variants \(Incremental, Kernel, Sparse, Mini Batch Sparse\)", dp.Group(
			dp.Text("### PCA"), dim_plots(pca_ex), 
			dp.Text("### Incremental PCA"), dim_plots(ipca_ex), 
			dp.Text("### Kernel PCA"), dim_plots(kpca_ex), 
			#dp.Text("### Sparse PCA"), dim_plots(spca_ex), 
			#dp.Text("### Mini Batch Sparse PCA"), dim_plots(minibatchspca_ex)
			)),
		content_group("Gaussian & Sparse Random Projection", dp.Group(
			dp.Text("### Gaussian Random"), dim_plots(gaussrandom_ex), 
			dp.Text("### Sparse Random"), dim_plots(sparserandom_ex))),
		content_group("t-Distributed Stochastic Neighbor Embedding", dim_plots(tsne_ex)),
		content_group("Factor Analysis", dim_plots(factor_ex)),
		content_group("Independent Component Analysis", dim_plots(fastica_ex)),
		content_group("Latent Dirichlet Allocation", dim_plots(lda_ex)),
		content_group("Nonnegative Matrix Factorization", dim_plots(nmf_ex))
		]),
	layout = dp.PageLayout.SIDE
	)

rprt.save(path = './outputs/embs_rprt.html', open=True)
rprt.upload(name= "The Ex-Academic's Sentence Embedding Guide", open = True, publicly_visible = True)
#https://datapane.com/reports/dkjbvwk/literature-in-bloom/