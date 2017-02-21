#coding=utf-8


'''this script is to get pro of user according to rule'''

import config
from condition import Condition
from condition import FieldDict
import pandas as pd
import numpy as np
import os
from fill import mc
from fill import util
import collections
from imblearn.over_sampling import ADASYN,SMOTE
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import merge


def process(rule):
	
	fd = FieldDict(config.fd_path)

	## get the new rule according to field 2 int map
	con = fd.parse(Condition(rule))

	df = pd.read_csv(config.i2int_path)

	## iterate the instance and get the labels --> Y
	labels = con.get_labels(df,np.nan)

	## get X
	columns = df.columns.values.tolist()
	columns.remove('User_name')
	names = list(df['User_name'])
	X = df[columns].as_matrix().astype(np.float64)

	## fill the X matrix
	filled_path = config.i2int_path + "_fill"
	if not os.path.exists(filled_path):
		X = mc.fill_knn_whole(X)
		np.savetxt(filled_path,X)
	else:
		X = np.loadtxt(filled_path,dtype=np.float64)

	## check whether the labels balance or not
	ld = dict(collections.Counter(labels))
	print ld
		
	## onehot
	enc = OneHotEncoder()
	newX = enc.fit_transform(X)
	newY = labels
	

	## isolate the unknown labels part first
	known_index,unknown_index,subX,subY = getSub(newX,newY)

	## add the unknown label part
	leftX = newX[unknown_index,:]
	leftY = [-1]*len(unknown_index)
	
	if not (ld.has_key(1)  and ld.has_key(0)):
		print 'the rule has no strictly fitable data'
		return 
	if util.isImBalance(ld):
		## imbalance data,oversample the data
		newX,newY = oversample(subX,subY)
		
		added = newX.shape[0] - subX.shape[0]		
		
		#index = mergeIndex(known_index,unknown_index,added)
		
	else:
		newX = subX
		newY = subY
		
	## do the pred
	result = semi_supervised(newX,newY,leftX,leftY)
	
	#rd = collections.Counter(list(result))
	#log_dict(rd)
	#np.savetxt('./tmp',result,fmt='%.2f')
	result = mergeResult(labels,result)
	tmp = [round(r,1) for r in result]
	log_dict(collections.Counter(tmp))
	return zip(names,result)

def mergeResult(labels,result):
	i = 0
	for index,l in enumerate(labels):
		if l == -1:
			labels[index] = result[i]
			i += 1
	return labels	

def log_dict(cd):
	cd = dict(cd)
	for key in cd:
		print key,cd[key]
	
def semi_supervised(newX,newY,leftX,leftY):
	if sp.issparse(newX): newX = newX.toarray()
	#model = LabelPropagation(n_jobs=3)
	model = LabelSpreading(n_jobs=3)
	model.fit(newX,newY)
	result = model.predict_proba(leftX)
	print model.score(newX,newY)
	return result[:,1].round(2)
		
def mergeIndex(known_index,unknown_index,added):
	lkn = len(known_index)
	luk = len(unknown_index)
	index = range(lkn + luk)
	i = 0
	j = 0
	k = 0
	while i < lkn and j< luk:
		if known_index[i] < unknown_index[j]:
			index[k] = i	
			i += 1
		else:
			index[k] = added + j
			j += 1
		k += 1
	while i < lkn:
		index[k] = i
		i += 1
		k += 1
	while j < luk:
		index[k] = added + j
		j += 1
		k += 1
	return index
	
def getSub(X,Y):
	known_index = [ index for index,l in enumerate(Y) if l in [0,1]]
	unknown_index= [index for index,l in enumerate(Y) if l == -1]
	subY = [Y[i] for i in known_index]
	subX = X[known_index,:]
	return known_index,unknown_index,subX,subY

def oversample(X,Y):
	#fm = ADASYN(random_state=42)
	print 'do the oversampling procedure'
	fm = SMOTE()
	newX,newY = fm.fit_sample(X.toarray(),Y)
	#newX = newX.astype(np.int).astype(np.float64)
	return newX,newY



if __name__ == '__main__':
	#rule =  {'Gender':'Male','Age':20,'Location':['California','Nebraska']}
	rule =  {'Gender':'Male'}
	
	pro = process(rule)
	## you may use merge.log_probs to save the probability
	merge.log_probs(pro,'./tmp')
	
	
