#coding=utf-8


'''this script is to get pro of user according to rule'''

import config
from condition import Condition
from condition import FieldDict
import pandas as pd
import numpy as np
import os
from fill import mc
import collections
from imblearn.over_sampling import ADASYN,SMOTE
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.preprocessing import OneHotEncoder


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
	index = range(X.shape[0])
	
	newX = X
	newY = labels

	if ld[1] < ld[0] / 2:
		## imbalance data,oversample the data
		known_index,unknown_index,subX,subY = getSub(X,labels)
		## isolate the unknown labels part first
		newX,newY = oversample(subX,subY)
		
		added = newX.shape[0] - subX.shape[0]
		
		## add the unknown label part
		leftX = X[unknown_index,:]
		leftY = [-1]*len(unknown_index)
		
		index = mergeIndex(known_index,unknown_index,added)
		
		
		newX = np.concatenate((newX,leftX),axis=0)
		newY = np.concatenate((newY,leftY),axis=0)

	## onehot
	enc = OneHotEncoder()
	newX = enc.fit_transform(newX)

	## do the pred
	semi_supervised(newX,newY)
	
def semi_supervised(newX,newY):
	label_prop_model = LabelPropagation()
	label_prop_model.fit(newX,newY)
	result = label_prop_model.predict_proba(newX)
	print result
		
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
	fm = ADASYN(random_state=42)
	#fm = SMOTE()
	newX,newY = fm.fit_sample(X,Y)
	newX = newX.astype(np.int).astype(np.float64)
	return newX,newY



if __name__ == '__main__':
	rule =  {'Gender':'Male','Age':20,'Location':['California','Nebraska']}
	
	process(rule)
	
	
