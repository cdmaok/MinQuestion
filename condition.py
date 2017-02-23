import pandas as pd
import numpy as np
import json,collections
from fill import util

class Condition:
	def __init__(self,rules):
		self.rules = rules 

	def getRule(self):
		return self.rules

	def extract(self,df):
		for key in self.rules:
			value = self.rules[key]
			if type(value) == list:
				df = df.ix[df[key].isin(value)]
			else:
				df = df.ix[df[key] == value]
		return df

	def get_labels(self,df,missing_value=np.nan,v = 0):
		labels = [getLabel(row,self.rules,missing_value,v) for index,row in df.iterrows()]
		return labels



def getLabel(row,rule,missing_value,v):
	'''
	1: yes, -1: unknown 0: no
	'''
	status = 2
	ans = []
	if v== 1 and row['User_name'] == '3l3NA':
		print 'test'
	for key in rule:
		value = rule[key]
		target = row[key]
		if util.isMissing(target,missing_value):
			status = -1
		else:
			if type(value) == list:
				status = 1 if row[key] in value else 0
			else:
				status = 1 if row[key] == value else 0
		ans.append(status)
	if 0 in ans:
		status = 0
	elif -1 in ans:
		status = -1
	else:
		status = 1
	if v == 1 and status == 1:
		print row['User_name'],ans
	return status
	

def hasKey(fd,value,missing_value):
	if type(missing_value) == str:
		return fd.has_key(value)
	elif np.isnan(missing_value):
		if type(value) == str:
			return fd.has_key(value)
		elif np.isnan(value):
			return True
		else:
			return fd.has_key(value)

def getValue(fd,key,missing_value):
	if type(missing_value) == str:
		return fd[key]
	elif np.isnan(missing_value):
		if type(key) == str:
			return fd[key]
		elif np.isnan(key):
			return 0
		else:
			return fd[key]

class FieldDict:
	
	def __init__(self,filename=None):
		self.fd = {}
		if filename == None: 
			print 'please load a fd file or train'
		else:
			self.load(filename)


	def train(self,filename,missing_value='?'):
		df = pd.read_csv(filename)
		fields = df.columns.values.tolist()[:]
		fields.remove('User_name')
		size = len(df[fields[0]].tolist())
		m = df.as_matrix()
		for f,v in enumerate(fields):
			f += 1
			self.fd[v] = {}
			for i in range(size):
				value = m[i][f]
				if util.isMissing(value,missing_value):continue
				if not hasKey(self.fd[v],value,missing_value):
					if type(value) == float: value = int(value)
					tmp = len(self.fd[v])+1
					self.fd[v][value] = tmp
				new_value= getValue(self.fd[v],value,missing_value)
				m[i][f] = new_value
		newdf = pd.DataFrame(data=m,columns=df.columns.values)
		newdf.to_csv(filename+'.2int',index=False)
		self.save(filename+'.fd')
		
	def save(self,filename):
		output = open(filename,'w')
		output.write(json.dumps(self.fd))
		output.close()
	
	def load(self,filename):
		line = open(filename).readline().strip()
		self.fd = json.loads(line)
	
	def parse(self,con):
		rules = con.getRule()
		tmp = {}
		for rule in rules:
			values = rules[rule]
			rule = unicode(rule)
			if type(values) == list:
				intlist = [ self.fd[rule][v] for v in values]
				tmp[rule] = intlist
			else:
				values = str(values)
				tmp[rule] = self.fd[rule][values]
		return Condition(tmp)
			
		

if __name__ == '__main__':
	#filename = '../mq_data/process_data/data.csv'
	filename = '../mq_data/process_data/data.csv.2int'
	rule =  {'Gender':'Male','Age':20,'Location':['California','Nebraska']}
	#rule =  {'Gender':'Male'}
	df = pd.read_csv(filename)
	#print df.describe()
	con = Condition(rule)
	#print con.extract(df)
	
	fdpath = '../mq_data/process_data/data.csv.fd'
	fd = FieldDict(fdpath)
	#fd = FieldDict()
	#fd.train(filename,missing_value = np.nan)

	newcon = fd.parse(con)
	l = newcon.get_labels(df,np.nan,v=1)
	print collections.Counter(l)
		
