#coding=utf-8

## Notice: PCA can only extract feature num which is the minimun of sample_num and feature num
import numpy as np
import threading,collections
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.feature_selection import RFE
from imblearn.over_sampling import ADASYN,SMOTE
import sampling_method
import math
from fill import util
import dt2

from skfeature.function.information_theoretical_based import MRMR
from sklearn.ensemble import AdaBoostClassifier


## get the first n elements of array,but if array[n-1] == array[n],then the (n+1)th element will be return.
def cut(score,index,size):
	src = score[index[size - 1]]
	total = len(score)
	ans = size - 1
	for i in range(size,total):
		tar = score[index[i]]
		if not isEqual(src,tar):
			break
		ans = i
	if size != ans: print 'expanding to' ,ans
	return ans
	
def isEqual(a1,a2):
	if abs(a1-a2)< 0.001:
		return True
	return False
	

def getHighScoreIndex(score,size):
	index  = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
	expand = cut(score,index,size)
	return index[0:expand]

class adaboostvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.boost()



	def boost(self):
		#print 'boost'
		x,y = getXY(self.sampled_df)
		clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
		clf.fit(x,y)
		score = clf.feature_importances_		
		t = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
		self.topics = t[:self.num]
		#score = list(score)
		#self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics
	
class MRMRvoter(threading.Thread):
	'''
	https://github.com/jundongl/scikit-feature/blob/master/skfeature/example/test_MRMR.py
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.feat = []



	def run(self):
		self.mrmr()



	def mrmr(self):
		
		x,y = getXY(self.sampled_df)
		feat = []
		idx = MRMR.mrmr(x,y, n_selected_features=self.num)
		for i in range(10):
			feat.append(idx[i])
		#print feat
		self.topics = feat
		#print self.topics



	
	def getTopic(self):
		return self.topics

	
class svmvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.svm()



	def svm(self):
		#print 'svm'
		x,y = getXY(self.sampled_df)
		clf = svm.SVC(kernel='linear')
		clf.fit(x,y)
		score = clf.coef_[0]
		score = list(score)
		self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics


class lassovoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.l1()



	def l1(self):
		#print 'lasso'
		x,y = getXY(self.sampled_df)
		#print x,y
		clf = linear_model.LogisticRegression(penalty='l1')
		clf.fit(x,y)
		score = clf.coef_[0]
		score = list(score)
		self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics

		
		
class dtvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.dt()



	def dt(self):
		#print 'dt'
		x,y = getXY(self.sampled_df)
		#clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=self.num)
		clf = tree.DecisionTreeClassifier(criterion='gini')
		clf.fit(x,y)
		score = clf.feature_importances_
		#print list(score)
		self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics


class Kbesetvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num,function=mutual_info_classif):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.func = mutual_info_classif

	def run(self):
		self.kbest()

	def kbest(self):
		x,y = getXY(self.sampled_df)
		kb = SelectKBest(self.func, k=self.num)
		kb.fit_transform(x, y)
		score = kb.scores_ 
		self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics

class VarianceVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.variance()


	def variance(self):
		x,y = getXY(self.sampled_df)
		v = list(np.var(x,axis=0))
		score = sorted(range(len(v)),key = lambda i:v[i],reverse=True)
		self.topics = getHighScoreIndex(score,self.num)
	
	def getTopic(self):
		return self.topics
		
		
class CorelationVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.corelation()


	def corelation(self):
		x,y = getXY(self.sampled_df)
		row,cols = x.shape
		v = [ np.corrcoef(x[:,c],y)[0,1] for c in range(cols)]
		score = sorted(range(len(v)),key = lambda i:v[i] if not math.isnan(v[i]) else -2,reverse=True)
		self.topics = getHighScoreIndex(score,self.num)
	
	def getTopic(self):
		return self.topics

class WrapperVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num,base=svm.SVC(kernel="linear",C=1)):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.base = base
	
	def run(self):
		self.rfe()


	def rfe(self):
		x,y = getXY(self.sampled_df)
		x,y = over_sampling(x,y)
		rfe = RFE(estimator=self.base,n_features_to_select=self.num)
		rfe.fit(x,y)
		self.topics = list(rfe.get_support(indices=True))

	
	def getTopic(self):
		return self.topics		

class WrapperDTVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num,base=tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
		self.base = base
	
	def run(self):
		self.rfe()


	def rfe(self):
		x,y = getXY(self.sampled_df)
		#x,y = over_sampling(x,y)
		rfe = RFE(estimator=self.base,n_features_to_select=self.num)
		rfe.fit(x,y)
		self.topics = list(rfe.get_support(indices=True))

	
	def getTopic(self):
		return self.topics		

class GBDTVoter(threading.Thread):
	
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num
	
	def run(self):
		self.gbdt()


	def gbdt(self):
		x,y = getXY(self.sampled_df)
		dt = GradientBoostingClassifier()
		dt.fit(x,y)
		#print dt.feature_importances_
		score = list(dt.feature_importances_)  
		self.topics = getHighScoreIndex(score,self.num)

	
	def getTopic(self):
		return self.topics			

def getXY(df):
	def replaceLabel(x):
		x = int(x)
		tmp = 1 if x == 4 else -1
		#tmp = 1 if x == 1 else -1
		return tmp		
	headers = list(df.columns)
	start = headers.index('user_topic')
	end = headers.index('Class')
	x = df.ix[:,start + 1:end].as_matrix()
	y = df.ix[:,end].apply(replaceLabel).as_matrix()

	return x,y
	
class RndLassovoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.l1()



	def l1(self):
		x,y = getXY(self.sampled_df)
		clf = linear_model.RandomizedLogisticRegression()
		clf.fit(x,y)
		score = list(clf.scores_)
		self.topics = getHighScoreIndex(score,self.num)


	
	def getTopic(self):
		return self.topics

class rfvoter(threading.Thread):
	'''
	http://scikit-learn.org/stable/modules/feature_selection.html
		
	'''
	def __init__(self,sampled_df,f_num):
		threading.Thread.__init__(self)
		self.sampled_df = sampled_df
		self.topics = []
		self.num = f_num



	def run(self):
		self.rf()



	def rf(self):
		x,y = getXY(self.sampled_df)
		x,y = over_sampling(x,y)
		clf = RandomForestClassifier(criterion='entropy',max_depth=self.num)
		clf.fit(x,y)
		score = clf.feature_importances_
		self.topics = getHighScoreIndex(score,self.num)

	
	def getTopic(self):
		return self.topics

def get_method(type=0):

	method_list = [svmvoter,lassovoter,dtvoter,Kbesetvoter,sampling_method.EntropyVoterSimple,VarianceVoter,CorelationVoter,WrapperVoter,RndLassovoter,GBDTVoter,rfvoter,dt2.DecisionTree,WrapperDTVoter,MRMRvoter,adaboostvoter]
	return method_list[type]
	
def over_sampling(x,y):
	ld = dict(collections.Counter(y))
	if len(ld) < 2:
		print 'warning,no enough data',ld
		print 'please add some more instance'
		return x,y
	else:
		if util.isImBalance(ld):
			print 'do the oversampling procedure'
			fm = ADASYN()
			return fm.fit_sample(x,y)
		else:
			return x,y
		

if __name__ == '__main__':
	#filename = ('./women_goal_fill.csv')
	filename = ('../mq_result/white_old_sim0_goal_origin_balan.csv')
	df = pd.read_csv(filename)
	#pv = Kbesetvoter(df,10)
	#pv.kbest()
	#pv = lassovoter(df,10)
	#pv.l1()
	#pv = svmvoter(df,10)
	#pv.svm()
	pv = dtvoter(df,10)
	pv.dt()
	#pv = VarianceVoter(df,10)
	#pv.variance()
	#pv = CorelationVoter(df,10)
	#pv.corelation()
	#pv = WrapperVoter(df,10)
	#pv.rfe()
	#pv = RndLassovoter(df,10)
	#pv.l1()
	#pv = GBDTVoter(df,10)
	#pv.gbdt()
	#pv = rfvoter(df,10)
	#pv.rf()
	print pv.getTopic()

