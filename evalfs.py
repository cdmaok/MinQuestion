#coding=utf-8
import os,merge,sys,classifier,config,collections,operator
import pandas as pd
import sampling_method,fs
import numpy as np
from fill import fill
from fill import mc
from fill import simrank
from knntext import text_sim
from sklearn.linear_model import LogisticRegression
from condition import Condition
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE


### this file is to check em fs 's performance
## sparse matrix single feature selection        dsfs
## matrix with svd single feature selection      mcsfs
## matrix with svd emsemble feature selection    mcemfs
data_path = config.data_path
result_path = config.result_path


rule = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}

## generate .origin file
def extract_matrix(userfile,vpfile):
	df = pd.read_csv(userfile)
	c = Condition({'Ethnicity':'White'})
	newdf = c.extract(df)
	#newdf.to_csv('../mq_exp/white_user',index = False)
	username = list(newdf['User_name'])
	df = pd.read_csv(vpfile)
	newdf = df.ix[df['user_topic'].isin(username)]
	newdf.to_csv('../mq_exp/white.origin',index=False)

def process_matrix(origin_file):
	df = pd.read_csv(origin_file)
	columns = getQuerylist(df)
	newdf = df.replace(['yes','no','?'],[1,-1,0])
	newdf = newdf.replace(['Republican Party','Democratic Party'],[1,-1])
	x = newdf[columns].as_matrix()
	y = newdf['Class'].as_matrix()
	newdf.to_csv('../mq_exp/white.matrix',index=False)

def process_mc(origin_file):
	df = pd.read_csv(origin_file)
	columns = getQuerylist(df)
	#f = mc.fill_knn_whole
	f = mc.fill_sim_whole
	newdf = mc.fill_whole(f,df)
	newdf = newdf.replace(['Republican Party','Democratic Party'],[1,-1])
	newdf.to_csv('../mq_exp/white.sim',index=False)

def getQuerylist(df):
	columns = df.columns.values.tolist()
	columns.remove('user_topic')
	columns.remove('Class')
	return columns
	

def getXY(df):
	columns = getQuerylist(df)
	x = df[columns].as_matrix()
	y = df['Class'].as_matrix()
	return x,y

def decide_feature_size(filename):
	df = pd.read_csv(filename)
	clf = LogisticRegression(penalty= 'l1')
	#clf = LinearSVC(penalty='l1',dual=False)
	x,y = getXY(df)
	ylist = list(y)
	counter = collections.Counter(ylist)
	print 'balance status:',counter
	x,y = fs.over_sampling(x,y)
	si = getTopicIndex(clf,x,y)
	newx = x[:,si]
	newy = y
	score = getScore(newx,newy)
	print 'score is ',score

def checkfs(filename):
	df = pd.read_csv(filename)
	clf = LinearSVC(penalty = 'l1',dual = False)
	ref = RFE(estimator = clf, n_features_to_select = 1)
	x,y = getXY(df)
	x,y = fs.over_sampling(x,y)
	ref.fit(x,y)
	ranking = ref.ranking_
	poslist = sorted(range(len(ranking)),key = lambda x: ranking[x])
	for i in range(1,len(ranking) + 1):
		ti = poslist[:i]
		newx = x[:,ti]
		newy = y
		score = getScore(newx,newy)
		print score

def getCVscore(x,y):
	accuracy = cross_val_score(clf, x,y, cv=5).mean()
        precision = cross_val_score(clf, x,y, cv=5, scoring='precision').mean()
        f1 = cross_val_score(clf, x,y, cv=5, scoring='f1').mean()
        recall = cross_val_score(clf, x,y, cv=5, scoring='recall').mean()
        roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc').mean()
	return accuracy,precision,f1,recall,roc_auc


def getScore(x,y):
	clf = LinearSVC(penalty='l2')
	## maybe rbf kernel
	result = []
	for i in range(10):
		clf.fit(x,y)
		score = clf.score(x,y)
		result.append(score)
	avg = sum(result) / float(len(result))
	return avg

def getTopicIndex(clf,x,y):
	x,y = fs.over_sampling(x,y)
	ref = RFE(estimator = clf, n_features_to_select = 1)
	ref.fit(x,y)
	ranking = ref.ranking_
	poslist = sorted(range(len(ranking)),key = lambda x: ranking[x])
	return poslist

def ensembleTopicIndex(ti,part,i):
	candidates = []
	for t in ti:
		candidates += t[:i]
	tc = dict(collections.Counter(candidates))
	final = []
	for key in tc:
		if tc[key] >= part / 2:
			final.append(key)
	return final

def ensembleTopicIndex2(ti,part,i):
	candidates = []
	for t in ti:
		candidates += t[:i]
	tc = dict(collections.Counter(candidates))
	sort_tc = sorted(tc.items(),key = operator.itemgetter(1),reverse = True)
	ans = [ e[0] for e in sort_tc]
	return ans
	
	
def checkemfs(col,totalti,part,x,y):
	for i in range(1,col+1):
		enti = ensembleTopicIndex2(totalti,part,i)
		if len(enti) < 1: 
			print 
			continue
		newx = x[:,enti]
		newy = y
		score = getScore(newx,newy)
		print score

def mcemfs(filename,part = 4):
	df = pd.read_csv(filename)
	x,y = getXY(df)
	rows,cols = df.as_matrix().shape
	per = rows / part
	start = 0
	end = start + per
	count = 0
	totalti = []
	clf = LinearSVC(penalty = 'l1',dual = False)
	while count < part - 1:
		print start,end
		px = x[start:end+1,:]
		py = y[start:end+1]
		ans = getTopicIndex(clf,px,py)
		totalti.append(ans)
		start = end + 1
		end = start + per
		count += 1
	px = x[start:,:]
	py = y[start:]
	#clf = LogisticRegression(penalty = 'l1')
	clf = LinearSVC(penalty='l1',dual=False)
	ans = getTopicIndex(clf,px,py)
	totalti.append(ans)
	checkemfs(cols,totalti,part,x,y)
		
def olemfs(filename,part = 4):
	df = pd.read_csv(filename)
	x,y = getXY(df)
	rows,cols = df.as_matrix().shape
	p1 = rows / part
	p2 = rows / ( part + 1)
	start = 0
	end = start + p2
	count = 0
	totalti = []
	clf = LinearSVC(penalty = 'l1',dual = False)
	while count < part - 1:
		print start,end
		px = x[start:end+1,:]
		py = y[start:end+1]
		ans = getTopicIndex(clf,px,py)
		totalti.append(ans)
		start += p1
		end = start + p2
		count += 1
	px = x[start:,:]
	py = y[start:]
	#clf = LogisticRegression(penalty = 'l1')
	clf = LinearSVC(penalty='l1',dual=False)
	ans = getTopicIndex(clf,px,py)
	totalti.append(ans)
	checkemfs(cols,totalti,part,x,y)

def bsemfs(filename,part = 4):
	df = pd.read_csv(filename)
	x, y = getXY(df)
	row,col = df.as_matrix().shape
	clf = LinearSVC(penalty = 'l1',dual = False)
	per = row / (part - 1)
	totalIndex = []
	for i in range(part):
		bsIndex = np.random.choice(row,per)
		newx = x[bsIndex,:]
		newy = y[bsIndex]
		newx,newy = fs.over_sampling(newx,newy)
		ans = getTopicIndex(clf,newx,newy)
		totalIndex.append(ans)
	checkemfs(col,totalIndex,part,x,y)

def adjustpl(pl,x,y,index,clf):
	row = x.shape[0]
	newx = x[index,:]
	newy = y[index]
	clf.fit(newx,newy)
	predicty = clf.predict(x)
	fit = [ True if y[i] == predicty[i] else False for i in range(row)]
	fc = dict(collections.Counter(fit))
	print fc
	total = fc[True] + fc[False] * 3
	pl = [ 1/ float(total) if fit[i] else 3 / float(total) for i in range(row)]
	return pl
	

def abemfs(filename,part = 4):
	
	df = pd.read_csv(filename)
	x,y = getXY(df)
	row,col = df.as_matrix().shape
	clf = LinearSVC(penalty = 'l1',dual = False)
	pl = [1/ float(row)] * row
	per = row / (part - 1)
	totalIndex = []
	for i in range(part):
		bsIndex = np.random.choice(row,per,p = pl)
		newx = x[bsIndex,:]
		newy = y[bsIndex]
		newx,newy = fs.over_sampling(newx,newy)
		ans = getTopicIndex(clf,newx,newy)
		totalIndex.append(ans)
		pl = adjustpl(pl,x,y,bsIndex,clf)
	checkemfs(col,totalIndex,part,x,y)


def main():
	filename = result_path + 'white_old'
	
	probs_file = filename + '.pro'

	fill_method_name = 'svd'
        threshold = 0 

	vote_matrix = 'topic_matric_origin.csv'
    
        origin_file = data_path + vote_matrix   
        vote_matrix = vote_matrix.replace('topic_matric','')
        origin_fill = filename + '_'+ fill_method_name + str(threshold) + vote_matrix
        goal_file = filename + '_goal' + vote_matrix 
	print 'using matrix file from', goal_file



if __name__ == '__main__':
	#extract_matrix('../mq_data/user_info_twoparty.csv','../mq_data/topic_matric_twoparty_balan_reduce.csv')
	#process_matrix('../mq_exp/white.origin')
	#process_mc('../mq_exp/white.origin')
	#mcemfs('../mq_exp/white.svd')
	olemfs('../mq_exp/white.matrix')
	#bsemfs('../mq_exp/white.matrix')
	#abemfs('../mq_exp/white.matrix')
	#decide_feature_size('../mq_exp/white.origin')
	#checkfs('../mq_exp/white.sim')
