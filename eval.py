#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:46:33 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

def eval_ch(emb, cl):
	return calinski_harabasz_score(emb, cl)

def eval_db(emb, cl):
	return davies_bouldin_score(emb, cl)
	
def eval_s(emb, cl):
	return silhouette_score(emb, cl)

