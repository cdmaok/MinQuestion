#coding=utf-8
import os
import sys
import random
import numpy as np
import condition
from sklearn.utils import resample
from pro import pro
from pro import pro_twoparty
import sampling_method
import pandas as pd
import math,numpy
import fs
import random

## this file is to merge get pro and sampling method script file

def get_sample_file(probs,df,frac=0.8,type = 0):
	# only extract people with pro == 1	

	df = pd.read_csv(df,index_col=0,dtype={"user_topic":str,"Class":str})
	if(type == 1): #resample
		users = list(df.index.values)
		size = int(len(users)*frac)
		#print size
		sampled_names = resample(users,n_samples=size)
		sample_df = df.ix[sampled_names,:]
		#sample_df.to_csv('test.csv')
	else:
		users = list(df.index.values)
		#print users
		size = int(len(users)*frac)
		#print size
		sampled_names = random.sample(users,size)
		sfile = '../mq_result/fill_data/sample/goal/sample_' + str(frac)
		f= open(sfile,'a')
		f.write(str(sampled_names)+'\n')
		f.close()


def sample(probs,df,t,frac=0.8,type = 0):
	# only extract people with pro == 1	
	sfile = '../mq_result/fill_data/sample/goal/sample_' + str(frac)
	sampled_all = [eval(e.strip()) for e in open(sfile).readlines() ]
	sampled_names = sampled_all[t]
	#print sampled_names
	df = pd.read_csv(df,index_col=0,dtype={"user_topic":str,"Class":str})
	sample_df = df.ix[sampled_names,:]
	users = list(sample_df.index)
	#print users == sampled_names
	labels = list(sample_df.drop_duplicates(subset='Class').Class)
	return sample_df,labels	
	
def get_sample(probs_file,origin_file,t,frac,type):
    #print origin_file
    probs = read_probs(probs_file)	
    df,labels = sample(probs,origin_file,t,frac,type)
    sampled_df = processdf(df,labels)
    #print(sampled_df['Class'].value_counts())
    return sampled_df		
	
def processdf(df,labels):
	def classmap(x):
		value = x['Class']
		index = labels.index(value)
		x['Class'] = index
		return x
	#df = df.replace('yes',1).replace('no',-1).replace('?',numpy.nan)
	df = df.replace('yes',1).replace('no',-1).replace('?',0)
	df = df.apply(classmap,axis=1)
	return df
	
def log_probs(probs,filename):
	output = open(filename,'w+')
	for e in probs:
		output.write(str(e) + '\n')
	output.close()
	
def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs

def get_probs_file(rule,probs_file,tp):
	con = condition.Condition(rule)
	if(tp):
		probs = [ e  for e in pro_twoparty.getPro(con.getRule()) if type(e[0]) != float]
	else:
		probs = [ e  for e in pro.getPro(con.getRule()) if type(e[0]) != float]
	log_probs(probs,probs_file)


def goal_all(_list,pos=0):	
	#print len(_list)
	_indexes = []
	i=0
	while i < len(_list):
		if(pos == 0):
			_indexes.append(_list[i][0])
		else:
			pro = _list[i][1]
			user_name = _list[i][0]
			if pro >= 0.5 and pro <= 1:
				_indexes.append(_list[i][0])
		i += 1
	print('the number of this group: ',len(_indexes))
	return _indexes

def goal_file(_list,_user):	
	#print len(_list)
	_indexes = []
	i=0	
	while i < len(_list):	
		pro = _list[i][1]
		user_name = _list[i][0]
		#print pro
		if pro == 1 and user_name in _user:
			#print 'ok'
			_indexes.append(_list[i][0])
		i += 1
	print('the number of goal group: ',len(_indexes))
	return _indexes	
	
def MergeTopic(probs,df,type,multi = True):
	# only extract people with pro == 1	
	if(type == 1):
		df = pd.read_csv(df)
		users = list(df.ix[:,'user_topic'])
		sampled_names = goal_file(probs,users)	
		
	elif(type == 2):
		sampled_names = goal_all(probs,1)
		
	labels = list(df.drop_duplicates(subset='Class').Class)
	condict = {'user_topic':sampled_names}
	if multi: condict['Class'] = ['Republican Party','Democratic Party']
	cons = condition.Condition(condict)
	return cons.extract(df),labels

	
def get_tp_file(probs_file,df,origin_file):
    probs = read_probs(probs_file)	
    sampled_df,labels = MergeTopic(probs,df,2)
    #print('the number of this group: ', sampled_df.shape)
    sampled_df.to_csv(origin_file,index=False)
    return sampled_df	
	
def get_goal_file(probs_file,goal_file,origin_file):	
	probs = read_probs(probs_file)
	df,labels = MergeTopic(probs,origin_file,1)
	df = processdf(df,labels)
	df.to_csv(goal_file,index=False)
	return df 
	
def main(df,num=10):
	tree = fs.dtvoter(df,num)
	tree.dt()

	lo = fs.lassovoter(df,num)
	lo.l1()
	
	svm = fs.svmvoter(df,num)
	svm.svm()


if __name__ == '__main__':
	#rule = {'Occupation':'Student'}
	#rule = {'Gender':'Female'}
	#rule = {'Ethnicity':'White','Age':['40','50','70','90','100']}
	#con = condition.Condition(rule)
	#probs = [ e  for e in pro.getPro(con.getRule()) if type(e[0]) != float]
	#filename = './women.pro'
	#log_probs(probs,filename)
	
	#probs = read_probs(filename)

	#originfile = './user_topic_origin.csv'
	#df,labels = MergeTopic(probs,originfile)
	#df = processdf(df,labels)
	csvname = '/home/yangying/mq_result/white_old_knn0_goal_twoparty_balan.csv'
	#df.to_csv(csvname)
	#csvname = './fs_file/fs_entropyvoter_en_200.csv'
	df = pd.read_csv(csvname)
	#print df
	main(df)
	
