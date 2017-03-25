#coding=utf-8
__author__ = 'llt'
from random import choice
import numpy as np
import pandas as pd
#from pro import label_propagation_accuracy
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
import sys
sys.path.insert(0,'..')
import config
result_path = config.result_path
data_path = config.pro_data_path
middle_0_path = config.pro_middle_0_path
middle_freq_path = config.pro_middle_freq_path
# data_path = './data.csv'
# middle_0_path = './middledata_0.csv'
# middle_freq_path ='./middledata_frequent.csv'
def attr(): ##çç‚´â‚¬Ñƒå½‡ï¿?
	df = pd.read_csv(data_path)
	mat = df.as_matrix()
	row,col = mat.shape
	attr = [[] for i in xrange(col)]
	for i in xrange(1,col):
		for j in xrange(row):
			if mat[j][i]==mat[j][i]:
				if mat[j][i] not in attr[i]:
					attr[i].append(mat[j][i])
		attr[i] = [x for x in attr[i] if attr[i].count(x) == 1]
	return attr,row,col

def getPro(rule):
	lattr,row,col = attr()
	attributes = ['User_name','Age','Location','Gender','President','Ideology','Education','Party','Ethnicity','Relationship','Income','Interested','Occupation','Looking','Religion']
	ind = [attributes.index(key) for key in rule]
	feature = [rule[key] if type(rule[key]) == list else [rule[key]]for key in rule]
	attr_num = len(rule)
	fea = [[] for i in xrange(attr_num)]
	for k in xrange(attr_num):
		for j in feature[k]:
			if j in lattr[ind[k]]:
				fea[k].append(lattr[ind[k]].index(j))

	line_num = 0
	Mat_Label_1 = []
	Mat_Label_0 = []
	Mat_Unlabel = []
	num_1 = 0
	num_0 = 0
	df_1 = pd.read_csv(middle_0_path)
	with open(middle_freq_path) as f:
		for line in f:
			line_num += 1
			if line_num != 1:
				line = line.strip('\n')
				line = line.split(',')
				for j in xrange(1,len(line)):
					line[j] = float(line[j])
				miss = 0
				for j in xrange(attr_num):
					if df_1.iloc[line_num-2,ind[j]] == 999: ##missing value
						miss = 1
				if miss == 1: ##missing value
					continue
				else:
					miss = 0
					tmp = []
					for j in xrange(attr_num):
						if line[ind[j]] in fea[j]: ##missing value
							miss += 1
						if j ==0:
							tmp.extend(line[1:ind[j]])
						else:
							tmp.extend(line[ind[j-1]+1:ind[j]])
					tmp.extend(line[ind[j]+1:col])
					if miss == attr_num:
						Mat_Label_1.append(tmp)
						num_1 += 1
					else:
						Mat_Label_0.append(tmp)
						num_0 += 1
	#print num_1,num_0

	return Mat_Label_0,Mat_Label_1,num_0,num_1

def accuracy(Mat_Label_0,Mat_Label_1,num_0,num_1):
	Mat_sample = Mat_Label_0[:]
	labels_sample = [0 for j in xrange(num_0)]
	Mat_sample.extend(Mat_Label_1[:])
	tmp = [1 for j in xrange(num_1)]
	labels_sample.extend(tmp)
	if num_1>num_0:
		num = num_0
		while num < num_1:
			x = choice(Mat_Label_0)
			Mat_sample.append(x)
			labels_sample.append(0)
			num += 1
	else:
		num = num_1
		while num < num_0:
			x = choice(Mat_Label_1)
			Mat_sample.append(x)
			labels_sample.append(1)
			num += 1

	Mat_Label = np.array(Mat_sample)
	labels = np.array(labels_sample)

	enc = OneHotEncoder()
	enc.fit(Mat_Label)
	Mat_Label = enc.transform(Mat_Label).toarray()

	precision_1 = []
	precision_2 = []
	recall_1 = []
	recall_2 = []
	f1_1 = []
	f1_2 = []
	methods = ['lpa','lr','svm','bayes','knn']

	cv = KFold(n_splits=10,shuffle=True).split(Mat_Label)
	for train_index, test_index in cv:
		p_1 = []
		p_2 = []
		r_1 = []
		r_2 = []
		f_1 = []
		f_2 = []
		for method in methods:
			if method == 'lpa':
				clf = LabelPropagation(kernel='rbf', gamma=20, n_neighbors=7,alpha=1, max_iter=30, tol=1e-3, n_jobs=1)
			elif method == 'lr':
				clf = LogisticRegression()
			elif method == 'svm':
				clf = svm.SVC()  # class
			elif method == 'bayes':
				clf = GaussianNB()
			elif method == 'knn':
				clf = KNeighborsClassifier()

			clf.fit(Mat_Label[train_index],labels[train_index])
			unlabel_data_labels = clf.predict(Mat_Label[test_index])

			p_1.append(metrics.precision_score(labels[test_index] ,unlabel_data_labels))
			r_1.append(metrics.recall_score(labels[test_index],unlabel_data_labels))
			f_1.append(metrics.f1_score(labels[test_index],unlabel_data_labels))

			clf.fit(Mat_Label[test_index],labels[test_index])
			unlabel_data_labels = clf.predict(Mat_Label[train_index])

			p_2.append(metrics.precision_score(labels[train_index] ,unlabel_data_labels))
			r_2.append(metrics.recall_score(labels[train_index],unlabel_data_labels))
			f_2.append(metrics.f1_score(labels[train_index],unlabel_data_labels))

		precision_1.append(p_1)
		recall_1.append(r_1)
		f1_1.append(f_1)
		precision_2.append(p_2)
		recall_2.append(r_2)
		f1_2.append(f_2)
	return  np.array(precision_1).mean(axis=0).tolist(),np.array(precision_2).mean(axis=0).tolist(),np.array(recall_1).mean(axis=0).tolist(),np.array(recall_2).mean(axis=0).tolist(),np.array(f1_1).mean(axis=0).tolist(),np.array(f1_2).mean(axis=0).tolist()
  
# main function
if __name__=="__main__":
	rule_1 = {'Education':['High School','Some College','Associates Degree']}
	rule_2 = {'Income':['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000']}
	rule_3 = {'Ethnicity':'White'}
	rule_4 = {'Gender':['Female'],'Age':[10,20,30]}
	rule_5 = {'Gender':['Male'],'Age':[10,20,30]}
	rule_6 = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}
	rule_7 = {'Education':['High School','Some College','Associates Degree'],'Income':['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000']}
	rule_8 = {'Ethnicity':'White','Age':[10,20,30]}
	rule_9 = {'Gender':['Male'],'Income':['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000']}
	rule_10 = {'Ethnicity':'White','Gender':['Male']}
	rule_11 = {'Party':['Democratic Party','Republican Party']}
	rule_12 = {'Age':[60,70,80,90,100,110]}
	rule_13 = {'Age':[30,40,50]}
	rule_14 = {'Occupation':['Retired']}
	rule_15 = {'Occupation':['Student']}
	rules = [rule_1,rule_2,rule_3,rule_4,rule_5,rule_6,rule_7,rule_8,rule_9,rule_10,rule_11,rule_12,rule_13,rule_14,rule_15]
	csvfile = file(result_path + '/p_r_f.csv','wb')
	writer = csv.writer(csvfile)

	data = []
	for rule in rules:
		Mat_Label_0,Mat_Label_1,num_0,num_1 = getPro(rule)
		method = ['lpa','lr','svm','bayes','knn']
		p1,p2,r1,r2,f1,f2 = accuracy(Mat_Label_0,Mat_Label_1,num_0,num_1)
		data.extend( [p1,p2,r1,r2,f1,f2,num_1,num_0])

	title = ['precisionï¼? train ; test =  9:1','precisionï¼? train ; test =  1:9','recallï¼? train ; test =  9:1','recallï¼? train ; test =  1:9','f1ï¼? train ; test =  9:1','f1ï¼? train ; test =  1:9']
	for i in xrange(len(title)):
		writer.writerow([title[i]])
		writer.writerow(['rule','lpa','lr','svm','bayes','knn','yes','no'])
		for j in xrange(len(rules)):
			tmp = [rules[j]]
			tmp.extend(data[8*j+i])
			tmp.append(data[8*j+6])
			tmp.append(data[8*j+7])
			writer.writerow(tmp)
		writer.writerow('\n')



