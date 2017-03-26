# -*- coding: utf-8 -*-
import pandas as pd 
import merge
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from knntext.doc2vec import getvector
import config
Filemodel = config.Filemodel



def predictability(col1,col2):
	l = len(col1)
	miss2 = 0
	vote = 0
	inter = 0
	union = 0
	for i in range(l):
		if col2[i] == '?':
			miss2 = miss2 + 1
		else:
			vote = vote + 1
			if col1[i] == col2[i]:
				inter = inter + 1
	
	pre = inter / float(vote)
	#print inter,vote,pre
	return pre
			

def relevant(col1_name,col2_name):
	vectors = [list(getvector(q,Filemodel)) for q in [col1_name,col2_name]]
	m = np.asarray(vectors)
	sim = cosine_similarity(m,m)
	return sim[0][1]



def pos_response_bias(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 
			if prob[i][1] == 1:
				group = group + 1
				if col1[i] == 'yes' :
					inter = inter + 1
				else: miss1 = miss1 + 1
	
	if not group == 0:
		pres = inter / float(group)
	else: pres = 0
	#print inter,group,pres
	return pres

def pos_response_bias_global(col1,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if prob[i][1] == 1:
			group = group + 1
			if col1[i] == 'yes' :
				inter = inter + 1
			else: miss1 = miss1 + 1
	
	if not group == 0:
		pres = inter / float(group)
	else: pres = 0	
	#print inter,group,pres
	return pres	
	
def response_bias(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 	
			if prob[i][1] == 1:
				group = group + 1
				if col1[i] == 'yes' or col1[i] == 'no':
					inter = inter + 1
				else: miss1 = miss1 + 1
	
	if not group == 0:
		res = inter / float(group)
	else: res = 0
	#print inter,group,res
	return res

def response_bias_global(col1,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if prob[i][1] == 1:
			group = group + 1
			if col1[i] == 'yes' or col1[i] == 'no':
				inter = inter + 1
			else: miss1 = miss1 + 1
	
	if not group == 0:
		res = inter / float(group)
	else: res = 0
	#print inter,group,res
	return res

def nonresponse_bias(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 	
			if prob[i][1] == 1:
				group = group + 1
				if col1[i] == '?':
					inter = inter + 1
				else: miss1 = miss1 + 1
	
	if not group == 0:
		nonres = inter / float(group)
	else: nonres = 0
	#print inter,group,nonres
	return nonres

def nonresponse_bias_global(col1,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if prob[i][1] == 1:
			group = group + 1
			if col1[i] == '?':
				inter = inter + 1
			else: miss1 = miss1 + 1
	
	if not group == 0:
		nonres = inter / float(group)
	else: nonres = 0	
	#print inter,group,nonres
	return nonres
	
def coverage_bias(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	vote = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 
			if col1[i] == '?':
				miss1 = miss1 + 1
			else:
				vote = vote + 1
				if prob[i][1] == 1:
					inter = inter + 1
	if not vote == 0:
		cov = inter / float(vote)
	else: cov = 0
	#print inter,vote,cov
	return cov

def coverage_bias_global(col1,prob):
	l = len(col1)
	miss1 = 0
	vote = 0
	inter = 0
	union = 0
	for i in range(l): 
		if col1[i] == '?':
			miss1 = miss1 + 1
		else:
			vote = vote + 1
			if prob[i][1] == 1:
				inter = inter + 1
	if not vote == 0:
		cov = inter / float(vote)
	else: cov = 0
	#print inter,vote,cov
	return cov	

def choose_query_pair(pair_file):
	qstat_file = '../mq_result/stat/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	c1 = 0
	c2 = 0
	candi_list = []
	goal_list = []
	for q in qstat:
		if q[3] >= 10 and q[3] < 100:
			print q
			candi_list.append(q[0])
			c1= c1+1
		elif q[3] > 100:
			print q
			goal_list.append(q[0])
			c2= c2+1
	print c1,c2
	print candi_list
	print goal_list	
	f = open(pair_file,'a')
	f.write(str(candi_list)+'\n')
	f.write(str(goal_list)+'\n')
	f.close()
	

#def preprocess():


def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs
	
def main():
	filename = '../mq_data/topic_matric_origin.csv'
	#filename = '../mq_result/stat/test_v.csv'
	df = pd.read_csv(filename,dtype={"user_topic":str,"Class":str})
	pair_file = '../mq_result/stat/query_pair'
	probs_file = '../mq_result/stat/stu.pro'
	#rule = {'Ethnicity':'White'}
	rule = {'Occupation':['Student']}
	if not os.path.exists(probs_file):
		merge.get_probs_file(rule,probs_file,False)
	#else: print 'already exist the pro file!'
	
	if not os.path.exists(pair_file):
		choose_query_pair(pair_file)
	#else: print 'already exist the pair file!'	
	'''
	testfile = '../mq_result/stat/test_l.log'
	aa = read_probs(testfile)
	print aa
	'''
	pair_ij = read_probs(pair_file)	
	prob = read_probs(probs_file)
	'''
	for i in range(1,6):
		for j in range(2,3):
			if not i == j :
				col1 = df.ix[:,i]
				col2 = df.ix[:,j]
				#col1_name = df.iloc[:,i].name
				#col2_name = df.iloc[:,j].name
				pre = predictability(col1,col2)
				#rel = relevant(col1_name,col2_name)
				pres = pos_response_bias(col1,col2,prob)
				res = response_bias(col1,col2,prob)
				nres = nonresponse_bias(col1,col2,prob)
				cov = coverage_bias(col1,col2,prob)
				
				gpres = pos_response_bias_global(col1,prob)
				gres = response_bias_global(col1,prob)
				gnres = nonresponse_bias_global(col1,prob)
				gcov = coverage_bias_global(col1,prob)
				
				print [pre,pres,res,nres,cov,gpres,gres,gnres,gcov]
	
	'''
	for j in pair_ij[1]:
		for i in pair_ij[0]:
			col1 = df.ix[:,i]
			col2 = df.ix[:,j]

			col1_name = df.iloc[:,i].name
			col2_name = df.iloc[:,j].name

			pre = predictability(col1,col2)
			rel = relevant(col1_name,col2_name)

			pres = pos_response_bias(col1,col2,prob)
			res = response_bias(col1,col2,prob)
			nres = nonresponse_bias(col1,col2,prob)
			cov = coverage_bias(col1,col2,prob)

			gpres = pos_response_bias_global(col1,prob)
			gres = response_bias_global(col1,prob)
			gnres = nonresponse_bias_global(col1,prob)
			gcov = coverage_bias_global(col1,prob)
			
			print [i,j,pre,rel,pres,res,nres,cov,gpres,gres,gnres,gcov]
			
	

if __name__ == '__main__':
	main()