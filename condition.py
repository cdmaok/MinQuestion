import pandas as pd
import numpy as np
import json

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
		for f in fields:
			self.fd[f] = {missing_value:0}
			for i in range(size):
				value = df[f][i]
				if not hasKey(self.fd[f],value,missing_value):
					tmp = len(self.fd[f])
					self.fd[f][value] = tmp
				df[f][i] = getValue(self.fd[f],value,missing_value)
		df.to_csv(filename+'.2int',index=False)
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
			if values == list:
				print 
			else:
				print 
			
		

if __name__ == '__main__':
	#filename = '../mq_data/process_data/data.csv'
	rule =  {'Gender':'Male','Age':20,'Location':['California','Nebraska']}
	#df = pd.read_csv(filename)
	#print df.describe()
	#con = Condition(rule)
	#print con.extract(df)
	
	filename = '../mq_data/process_data/data.csv'
	#fdpath = '../mq_data/process_data/data.fd'
	#fd = FieldDict(fdpath)
	fd = FieldDict()
	fd.train(filename,missing_value = np.nan)
	#print fd.parse(con)
		
