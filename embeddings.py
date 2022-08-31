#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:45:53 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import pandas as pd
import sklearn as sk
import sklearn.feature_extraction
import streamlit as st
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer as snt

#preprocessing is handled outside of model as much as possible, including lowercase, tokenization, stopword removal, and other normalization functions

@st.cache(persist = True)
def model_tfidf(in_text, ngram_range):
	"""
TfidfVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)[source]¶
	"""
	return sk.feature_extraction.text.TfidfVectorizer(lowercase = False, ngram_range = ngram_range).fit_transform(in_text).toarray()

@st.cache(persist = True)
def model_hash(in_text, ngram_range):
	"""
HashingVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', n_features=1048576, binary=False, norm='l2', alternate_sign=True, dtype=<class 'numpy.float64'>)[source]¶
	"""
	return sk.feature_extraction.text.HashingVectorizer(lowercase = False, ngram_range = ngram_range).fit_transform(in_text).toarray()

@st.cache(persist = True)
def model_count(in_text, ngram_range):
	"""
CountVectorizer(*, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)[source]¶
	"""
	return sk.feature_extraction.text.CountVectorizer(lowercase = False, ngram_range = ngram_range).fit_transform(in_text).toarray()

@st.cache(persist = True)
def model_use(in_text):
	model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
	return model(in_text).numpy()

@st.cache(persist = True)
def model_snt(in_text, model_name):
	#https://www.sbert.net/docs/pretrained_models.html
	model = snt(model_name)
	return model.encode(in_text)

st_available_models = ["all-MiniLM-L12-v1","all-MiniLM-L12-v2","all-MiniLM-L6-v1","all-MiniLM-L6-v2","all-distilroberta-v1","all-mpnet-base-v1","all-mpnet-base-v2","all-roberta-large-v1","average_word_embeddings_glove.6B.300d","average_word_embeddings_komninos","distiluse-base-multilingual-cased-v1","distiluse-base-multilingual-cased-v2","gtr-t5-base","gtr-t5-large","gtr-t5-xl","gtr-t5-xxl","msmarco-bert-base-dot-v5","msmarco-distilbert-base-tas-b","msmarco-distilbert-dot-v5","multi-qa-MiniLM-L6-cos-v1","multi-qa-MiniLM-L6-dot-v1","multi-qa-distilbert-cos-v1","multi-qa-distilbert-dot-v1","multi-qa-mpnet-base-cos-v1","multi-qa-mpnet-base-dot-v1","paraphrase-MiniLM-L12-v2","paraphrase-MiniLM-L3-v2","paraphrase-MiniLM-L6-v2","paraphrase-TinyBERT-L6-v2","paraphrase-albert-small-v2","paraphrase-distilroberta-base-v2","paraphrase-mpnet-base-v2","paraphrase-multilingual-MiniLM-L12-v2","paraphrase-multilingual-mpnet-base-v2","sentence-t5-base","sentence-t5-large","sentence-t5-xl","sentence-t5-xxl"]
