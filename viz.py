#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 06:46:29 PM EDT 2022 
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import numpy as np
import pandas as pd
import plotly.express as px
import random

from cluster import *

random.seed(42)

px.defaults.template = "plotly"

def d2_plot(in_df):
	fig = px.scatter(
		x = np.array(in_df["d2"].values.tolist())[:,0], 
		y = np.array(in_df["d2"].values.tolist())[:,1], 
		color = list(map(str, in_df["cluster"])),
		hover_name = in_df["prep"]
		)
	fig.update_layout(showlegend=False)
	return fig

def d3_plot(in_df):
	fig = px.scatter_3d(
		x = np.array(in_df["d3"].values.tolist())[:,0], 
		y = np.array(in_df["d3"].values.tolist())[:,1], 
		z = np.array(in_df["d3"].values.tolist())[:,2], 
		color = list(map(str, in_df["cluster"])),
		hover_name = in_df["prep"]
		)
	fig.update_traces(marker={'size': 2})
	fig.update_layout(showlegend=False)
	return fig

def viz_ex(func, cldata, use_embs):
	dim_df = pd.DataFrame({
	"prep" : cldata,
	"cluster" : cluster_hdbscan(use_embs, 1.0, "euclidean", 5),
	"emb" : list(use_embs),
	"d2" : list(func[0]),
	"d3" : list(func[1])
	})
	out = {"d2": d2_plot(dim_df), "d3":d3_plot(dim_df)}
	return out