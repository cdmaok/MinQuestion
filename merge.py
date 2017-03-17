#coding=utf-8
import os
import sys
import random
import numpy as np
import condition
from pro import pro
from pro import pro_twoparty
import sampling_method
import pandas as pd
import math,numpy
import fs


## this file is to merge get pro and sampling method script file


def accept_sampling(_list,_size = 0):
	def decide_size(_list):
		size = 0
		count = 0
		for e in _list:
			if e[1] == 1:
				size += 1
			elif e[1] < 0.5 and e[1] > 0:
				count += 1
		return size + count * 0.8
		
	if _size == 0:
		# generate size automatically
		_size = decide_size(_list)
	_indexes = []
	length = len(_list)
	i = 0
	while i < _size:
		random_index = random.randint(0,length - 1)
		pro = _list[random_index][1]
		acc = random.uniform(0,1)
		if acc < pro:
			i += 1
			_indexes.append(_list[random_index][0])
	return _indexes

def pos_sampling(_list,_size = 0):
	def decide_size(_list):
		size = 0
		count = 0
		for e in _list:
			if e[1] <= 1 and e[1] > 0.5:
				size += 1
		return size * 0.8
		
	if _size == 0:
		# generate size automatically
		_size = decide_size(_list)
	_indexes = []
	length = len(_list)
	i = 0
	while i < _size:
		random_index = random.randint(0,length - 1)
		pro = _list[random_index][1]
		acc = random.uniform(0.5,1)
		if acc < pro:
			i += 1
			_indexes.append(_list[random_index][0])
	return _indexes	

	
	
	
def one_sampling(_list,_size = 0):
	probs_file2 = '../mq_result/two_party.pro'
	probs2 = read_probs(probs_file2)	
	
	_indexes = []
	tmp = []
	def decide_size(_list):
		size = 0
		count = 0
		for j in range(len(_list)):
			e = _list[j]
			#print e[1],probs2[j][1]
			if e[1] == 1  and probs2[j][1] == 1:
				_indexes.append(e[0])
				size += 1
		return int(size *0.8)
	#print tmp	
	if _size == 0:
		# generate size automatically
		_size = decide_size(_list)
	#print len(_list)
	#print('the number of this group: ',_size)

	sample = random.sample(_indexes,_size)
	#print _indexes
	#print sample	
	return sample
	
def goal_file(_list):	
	#print len(_list)
	_indexes = []
	i=0
	while i < len(_list):	
		pro = _list[i][1]
		#print pro
		if pro == 1:
			#print 'ok'
			_indexes.append(_list[i][0])
		i += 1
	print('the number of this group: ',len(_indexes))
	return _indexes
	
def goal_all(_list):	
	#print len(_list)
	_indexes = []
	i=0
	while i < len(_list):	
		_indexes.append(_list[i][0])
		i += 1
	return _indexes
	
	
def MergeTopic(probs,filename,multi = True):
	#sampled_names = accept_sampling(probs)
	#sampled_names = pos_sampling(probs)
	sampled_names = one_sampling(probs)
	df = pd.read_csv(filename)
	labels = list(df.drop_duplicates(subset='Class').Class)
	condict = {'user_topic':sampled_names}
	if multi: condict['Class'] = ['Republican Party','Democratic Party']
	cons = condition.Condition(condict)
	return cons.extract(df),labels

def MergeTopic2(probs,filename,multi = True):
	# only extract people with pro == 1
	sampled_names = goal_file(probs)
	#sampled_names = goal_all(probs)	
	df = pd.read_csv(filename)	
	labels = list(df.drop_duplicates(subset='Class').Class)
	condict = {'user_topic':sampled_names}
	if multi: condict['Class'] = ['Republican Party','Democratic Party']
	cons = condition.Condition(condict)
	return cons.extract(df),labels	
	
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

def preprocess(filename,goalname):
	
	df = pd.read_csv(filename)
	#df = df.replace('yes',1).replace('no',-1).replace('?',0)	
	n = len(df.iloc[0])-1
	print(n)
	col = [0]
	for i in range(1,n):
		b = df.iloc[:,i].name
		vote = list(df.drop_duplicates(subset=b).iloc[:,i])
		#print(vote)
		if('yes' in vote and 'no' in vote):
			col.append(i)
	col.append(-1)
	#print(col)	
	df = df.ix[:,col]
	print(len(df.iloc[0]))
	df.to_csv(goalname,index=False)
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

def get_file(probs_file,origin_file):
    probs = read_probs(probs_file)	
    df,labels = MergeTopic(probs,origin_file)
    sampled_df = processdf(df,labels)
    #print(sampled_df['Class'].value_counts())
    return sampled_df	
	
def get_goal_file(probs_file,goal_file,origin_file):	
	probs = read_probs(probs_file)
	df,labels = MergeTopic2(probs,origin_file)
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
	
