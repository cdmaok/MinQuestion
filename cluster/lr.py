# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def LRClassifier(x,y):
	print 'LR'
	#x,y = getXY(sampled_df)
	#clf = LogisticRegression(max_iter= 5000)
	#clf = SGDClassifier(loss = 'log',penalty='l1',n_iter= 5000)
	clf = SGDClassifier(n_iter= 10000)
	clf.fit(x,y)
	score = clf.coef_[0]
	score = list(score)
	print score
	
	accuracy = cross_val_score(clf, x,y, cv=5)
	precision = cross_val_score(clf, x,y, cv=5, scoring='precision')
	f1 = cross_val_score(clf, x,y, cv=5, scoring='f1')
	recall = cross_val_score(clf, x,y, cv=5, scoring='recall')
	roc_auc = cross_val_score(clf, x,y, cv=5, scoring='roc_auc')
	
	print np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(f1),np.mean(roc_auc)
	
	
def getXY(filename):
	def replaceLabel(x):
		x = int(x)
		tmp = 1 if x == 1 else 0
		return tmp
		
	data = pd.read_table(filename,sep = ',')
	print data.shape
	end = data.shape[1] -1
	x = data.ix[:,:end].as_matrix()
	y = data.ix[:,end].apply(replaceLabel).as_matrix()
	print x,y
	return x,y	

	
	
def main():
	filename = '/home/yangying/MinQuestion/cluster/tmp_p'
	
	x,y = getXY(filename)
	
	LRClassifier(x,y)

if __name__=='__main__':
	main()