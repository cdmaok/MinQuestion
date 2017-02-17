#coding=utf-8
__author__ = 'Administrator'

import pandas as pd
import numpy as np
import csv
import os
import json
import knntext

Comment_path = '/home/yangying/mq_data/comments'
Text_path = '/home/yangying/mq_data/text.csv'
Matric_path = '/home/yangying/mq_data/topic_matric_origin.csv'
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
	df = df.replace('yes','1').replace('no','0').replace('?',np.nan)
	m = df[cols].as_matrix().astype(np.float32)
	m = f(m)
	questions = df.columns.values.tolist()
	a = df['user_topic'].as_matrix()[:,None]
	b = df['Class'].as_matrix()[:,None]
	filled = np.concatenate((a,m,b),axis=1)
	newdf = pd.DataFrame(data=filled,columns=questions)
	return newdf

def fill_knn_whole(matrix):
	'''
	fill the matrix with knn
	'''
	# matrix = KNN(k=3).completes(matrix)
	matrix = knntext.KNN(k=3).complete(matrix)
	return matrix


def context(querylist,username):
	directory = Comment_path
	toadd = {}
	for filename in os.listdir(directory):
		filepath = directory + '/' + filename
		lines = [l.strip().decode('utf-8','ignore') for l in open(filepath).readlines()]
		for l in lines:
			l = l.replace('\\','/')
			dic = json.loads(l,strict= False)
			title = dic[u'title'].strip().encode('utf-8')
			text = dic[u'text'].strip().encode('utf-8')
			user = dic[u'name'].strip().encode('utf-8')
			if title in querylist and user in username:
				if user in toadd:
						tmp = toadd[user]
						tmp += title + ' ' + text
						toadd[user] = tmp
				else:
						toadd[user] = title + ' ' + text
	return toadd

def text(df):
	row,col = df.shape
	f = df.iloc[:,0]
	user = []
	for k in xrange(row):
		user.append(f.iloc[k])
	querylist = df.columns.values[1:-1]  #第一行
	querylist = [q.strip().decode('utf-8','ignore')  for q in list(querylist)]
	text = context(querylist,user)
	csvfile = file(Text_path,'wb')
	writer = csv.writer(csvfile)
	writer.writerow(['username','text'])
	for i in xrange(len(user)):
		if user[i] in text:
			writer.writerow([user[i],text[user[i]]])
		else:
			writer.writerow([user[i]])


if __name__ == '__main__':
	# df = pd.read_csv(Matric_path)
	# text(df)


	df = pd.read_csv(Matric_path)
	newdf = fill_whole(fill_knn_whole,df)
	# newdf.to_csv('./test_origin.csv',index=False)
	print newdf.as_matrix().shape

