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
	print "matrix shape",ndarray.shape
	row,col,edges = array2edge(ndarray)
	G = nx.DiGraph()

	G.add_nodes_from(range(row), bipartite=0)
	G.add_nodes_from(range(col), bipartite=1)

	G.add_edges_from(edges)
	m = gs.simrank_bipartite(G)
	print "similiarty",m.shape
	return m
	


def array2edge(nd):
	row,col = nd.shape
	edges = []
	for i in range(row):
		for j in range(col):
			if not util.isMissing(nd[i][j],np.nan):
				edges.append((i,j))
	return row,col,edges

if __name__ == '__main__':
	m = simrank_from_file('../../mq_data/topic_matric_twoparty_balan.csv')
	print m.shape
