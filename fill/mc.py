#coding=utf-8

## this code is demo about matrix completion
import pandas as pd
import numpy as np
from fancyimpute import MatrixFactorization,MICE,BiScaler, KNN, NuclearNormMinimization, SoftImpute,IterativeSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback


def fill_whole(f,df):
	'''
	fill the matrix with fancyimpute
	parameters notation:
	df : dataframe
	f: matrix completion function
	including fill_knn_whole, fill_nnm_whole,fill_sim_whole,fill_svd_whole,fill_svd_whole,fill_mice_whole,fill_mf_whole,fill_biscaler_whole
	'''
	cols = df.columns.values.tolist()
	cols.remove('user_topic')
	cols.remove('Class')
	df = df.replace('yes','1').replace('no','-1').replace('?',np.nan)
	m = df[cols].as_matrix().astype(np.float64)
	m = f(m)
	evalMatrix(m)
	questions = df.columns.values.tolist()
	a = df['user_topic'].as_matrix()[:,None]
	b = df['Class'].as_matrix()[:,None]
	filled = np.concatenate((a,m,b),axis=1)
	newdf = pd.DataFrame(data=filled,columns=questions)
	return newdf

def evalMatrix(m):
	m = np.reshape(m,(1,-1))
	print np.amin(m),np.amax(m)

def fill_knn_whole(matrix):
	'''
	ok with 1/-1
	fill the matrix with knn
	'''
	matrix = KNN(k=3,use_argpartition=True).complete(matrix)
	return matrix


def fill_nnm_whole(matrix):
	'''
	ok with 1/-1
NuclearNormMinimization: Simple implementation of Exact Matrix Completion via Convex Optimization by Emmanuel Candes and Benjamin Recht using cvxpy. Too slow for large matrices.
	'''
	matrix = NuclearNormMinimization().complete(matrix)
	return matrix

def fill_sim_whole(matrix):
	'''
	ok with 1/-1
SoftImpute: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the softImpute package for R, which is based on Spectral Regularization Algorithms for Learning Large Incomplete Matrices by Mazumder et. al.
	'''
	matrix = SoftImpute().complete(matrix)
	return matrix

def fill_svd_whole(matrix):
	'''
IterativeSVD: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from Missing value estimation methods for DNA microarrays by Troyanskaya et. al.
	'''
	matrix = IterativeSVD().complete(matrix)
	return matrix

def fill_mice_whole(matrix):
	'''
MICE: Reimplementation of Multiple Imputation by Chained Equations.
	'''
	matrix = MICE().complete(matrix)
	return matrix

def fill_mf_whole(matrix):
	'''
	ok with 1/-1
MatrixFactorization: Direct factorization of the incomplete matrix into low-rank U and V, with an L1 sparsity penalty on the elements of U and an L2 penalty on the elements of V. Solved by gradient descent.
	'''
	matrix = MatrixFactorization(verbose=False).complete(matrix)
	return matrix


def fill_biscaler_whole(matrix):
	'''
	ok with 1/-1
BiScaler: Iterative estimation of row/column means and standard deviations to get doubly normalized matrix. Not guaranteed to converge but works well in practice. Taken from Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares.
	'''
	matrix[np.isnan(matrix)] = 0
	matrix = BiScaler().fit_transform(matrix)
	return matrix
		



if __name__ == '__main__':
	filename = '../../mq_data/topic_matric_twoparty_balan.csv'
	df = pd.read_csv(filename)
	#newdf = fill_whole(fill_biscaler_whole,df)
	#newdf = fill_whole(fill_knn_whole,df)
	#newdf = fill_whole(fill_sim_whole,df)
	#newdf = fill_whole(fill_svd_whole,df)
	newdf = fill_whole(fill_mf_whole,df)
	try:
		newdf.to_csv('./test.csv',encoding='utf-8')
		print 'done'
	except:
		traceback.print_exc()
	## blahblah


