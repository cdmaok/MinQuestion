#coding=utf-8
import pandas as pd
import copy
import random
import os
import csv
from sklearn.metrics import mean_squared_error
from fill import fill
from fill import mc
from knntext import text_sim
from math import sqrt
import config

data_path = config.data_path
result_path = config.result_path

def reduce_mat(df):
	'''
	reduce the matrix with some answers
	'''
	query = df.columns.values[:]
	origin_matrix = df.as_matrix()
	reduce_matrix = copy.copy(origin_matrix)
	reduce = []
	for iter_i,i in enumerate(origin_matrix):
		for iter_j,j in enumerate(i):
			if j=='yes' or j=='no':
				reduce.append([iter_i,iter_j,j])
	slice = random.sample(reduce, 100)
	for i in slice:
		reduce_matrix[i[0],i[1]] = '?'
	df = pd.DataFrame(reduce_matrix,columns=query)
	df.to_csv(data_path + 'topic_matric_twoparty_balan_reduce.csv',index=False)
	return reduce_matrix,slice



def fill_matrix(origin_file,origin_fill,fill_method_name,threshold=0.6):
	df = pd.read_csv(origin_file)
	if fill_method_name == 'knn_rake' :
		fill_method = 'rake'
		df = fill.fill(rule,df,fill_method,threshold)
		fill_method = mc.fill_knn_whole
		df = mc.fill_whole(fill_method,df)
	elif fill_method_name == 'knn_doc' :
		fill_method = 'doc2vec'
		df = fill.fill(rule,df,fill_method,threshold)
		fill_method = mc.fill_knn_whole
		df = mc.fill_whole(fill_method,df)
	elif fill_method_name == 'knn_simrank' :
		#change config.py simrank_flag = True before use this method
		fill_method = text_sim.fill_knn_whole
		df = text_sim.fill_whole(fill_method,df,simf='simrank')
	elif fill_method_name == 'knn_text':
		fill_method = text_sim.fill_knn_whole
		df = text_sim.fill_whole(fill_method,df,simf='text')
	elif fill_method_name == 'itersvd' :
		fill_method = mc.fill_svd_whole
		df = mc.fill_whole(fill_method,df)
	elif fill_method_name == 'soft_impute' :
		fill_method = mc.fill_sim_whole
		df = mc.fill_whole(fill_method,df)
	elif fill_method_name == 'mf' :
		fill_method = mc.fill_mf_whole
		df = mc.fill_whole(fill_method,df)
	elif fill_method_name == 'biscaler':
		fill_method = mc.fill_biscaler_whole
		df = mc.fill_whole(fill_method,df)
	else:
		print 'Error: not input fill method'
	df.to_csv(origin_fill,index=False)
	print 'fill ok'



if __name__ == '__main__':

	rule = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}
	threshold = 0.6
	filename = result_path + 'white_old'
	vote_matrix = 'twoparty_balan_reduce.csv'
	fill_method = ['knn_text','knn_rake','knn_doc','knn_simrank','itersvd','soft_impute','mf','biscaler']
	total = [0 for i in xrange(len(fill_method))]
	#fill_method = ['knn']
	result = []
	for i in xrange(1):
		origin_file = data_path + 'topic_matric_twoparty_balan.csv'
		df = pd.read_csv(origin_file)
		reduce_matrix,slices = reduce_mat(df)
		value = []
		for j in slices:
			if j[2] == 'yes':
				value.append(1.)
			else:
				value.append(-1.)
		tmp = []
		for index,method in enumerate(fill_method):
			fill_method_name = method
			origin_fill = filename + '_'+ fill_method_name + str(threshold) + vote_matrix
			origin_file = data_path + 'topic_matric_twoparty_balan_reduce.csv'
			fill_matrix(origin_file,origin_fill,fill_method_name,threshold)

			df = pd.read_csv(origin_fill)
			mat = df.as_matrix()

			fill_value = []
			for k in xrange(len(value)):
				fill_value.append(mat[slices[k][0],slices[k][1]])
			rmse = sqrt(mean_squared_error(value,fill_value))
			tmp.append(rmse)
			total[index] += rmse
		#print tmp
		result.append(tmp)
	print result

	for i in xrange(len(fill_method)):
		print fill_method[i]
		print 'RMSE:'
		print total[i]/10.








