#coding=utf-8

import numpy as np
from collections import Counter

def count_sparsity(matrix,v = 0 ):
	'''
		count matrix sparsity.
	'''
	f = NoneElement()
	result = f(matrix).sum()
	if v == 1:
		print matrix.shape
		print result
	return 1 - result / float(np.prod(matrix.shape))

def isNone(value):
	if value == '?' or value == '0' or value == 0:
		return True
	return False

def most_commom(elements):
	d = Counter(elements)
	ele = d.most_common(1)[0][0]
	if isNone(ele):
		return None
	else:	
		return ele
	
def isMissing(value,missing_value):
        if type(missing_value) == str:
                return missing_value == value
        elif np.isnan(missing_value):
                if type(value) == str:
                        return False
                elif np.isnan(value):
                        return True
                else:
                        return missing_value == value

	

def NoneElement():
	return np.vectorize(isNone)


def sysmbol(vote):
	if vote == 'yes':
		return 1
	elif vote == 'no':
		return -1
	elif vote == '?':
		return 0
	elif vote > 0:
		return 1
	else:
		return 0


def getKey(node1,node2):
	if node1 < node2:
		return str(node1) + ',' + str(node2)
	else:
		return str(node2) + ',' + str(node1)

def parse(v):
	if v == 'yes':
		return 1
	elif v == 'no':
		return -1
	elif v == '?':
		return None
	elif int(v) > 0:
		return 1
	else:
		return -1

def isImBalance(ld):
	keys = ld.keys()
	major = ld[keys[0]]
	minor = ld[keys[1]]
	if major < minor:
		tmp = minor
		minor = major
		major = tmp
	ratio = float(minor) / major
	return ratio < 0.8

class ValueDict:
	
	def __init__(self,match,dt):
		self.match = match
		self.dt = dt
	
	def getFloat(self,node,indexs,values):
		values = list(values)
		ans = 0
		for key,index in enumerate(indexs):
			polar = self.match[getKey(node,index)]
			ans += sysmbol(values[key]) * polar
		if ans != 0:
			ans /= len(indexs)
			return ans
		else:
			return None
	
	def isSame(self,v1,v2):
		if self.dt == 'binary':
			return v1 == v2
		## float now
		return parse(v1) == parse(v2)


if __name__ == '__main__':
	m = np.zeros((2,2))
	print count_sparsity(m)
