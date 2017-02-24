#coding=utf-8


## this code is to generate similar matrix by simrank

import graphsim as gs
import networkx as nx
import numpy as np
import util
import pandas as pd


def simrank_from_file(filename):
	df = pd.read_csv(filename)
	cols = df.columns.values.tolist()
	cols.remove('user_topic')
	cols.remove('Class')
	df = df.replace('yes','1').replace('no','0').replace('?',np.nan)
	m = df[cols].as_matrix().astype(np.float64)
	return simrank(m)
	


def simrank(ndarray):
	edges = array2edge(ndarray)
	G = nx.DiGraph()
	#print edges
	G.add_edges_from(edges)
	print gs.simrank_bipartite(G)


def array2edge(nd):
	row,col = nd.shape
	edges = []
	for i in range(row):
		for j in range(col):
			if not util.isMissing(nd[i][j],np.nan):
				edges.append((i,j))
	return edges

if __name__ == '__main__':
	simrank_from_file('../../mq_data/topic_matric_twoparty_balan.csv')
