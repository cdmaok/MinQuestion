#coding=utf-8

### this script is to analyse the exp data
import config
import sys
import pandas as pd 

## demo -->4 repub --> 1

def get_fields(array):
	indexs = []
	votes = []
	texts = []
	for q in array:
		fs = q.split('+')
		indexs.append(fs[0])
		votes.append(fs[1])
		texts.append(fs[2])
	return indexs,votes,texts


def get_querys(filename):
	res = []
	f = open(filename)
	while True:
		line = f.readline()
		if not line: break
		if line.startswith('-------print feature'):
			line = f.readline().strip()
			querys = []
			while not line.startswith('['):
				querys.append(line)
				line = f.readline().strip()
			a,b,c = get_fields(querys)
			res.append('\n'.join(c))
	return res

def get_stat(filename):
	f = open(filename)
	whole = []
	while True:
		line = f.readline()
		if not line: break
		if line.startswith('SVM'):
			line = f.readline().strip()
			querys = []
			while not line.startswith('------'):
				querys.append(line)
				line = f.readline().strip()
				if not line: break
			querys = [ t.split() for i,t in enumerate(querys) if i not in [1,3,5,7]]
			whole.append(querys)
	row = 0
	col = 0
	stats = [size[row][col] for size in whole]
	return stats
	'''
	for size in whole:
		m = []
		#print size
		m = [size[row][i] for i in range(5)]
		#print s,' '.join(m)
		s += 10
	'''

def best(filename):
	q = get_querys(filename)
	s = get_stat(filename)
	g = sorted(range(len(s)),key = lambda x: s[x],reverse = True)
	print 'max precision is ', s[g[0]]
	return q[g[0]]

def extract_data(logfile,datafile=None):
	if datafile == None:
		print 'need datafile name'
		sys.exit()
	querys = best(logfile).split('\n')
	df = pd.read_csv(datafile)
	### Class
	namelist = df['user_topic'].tolist()
	columns = df.columns.values.tolist()
	columns = [col.strip() for col in columns]
	ndf = pd.DataFrame(data=df.as_matrix(),columns=columns)
	#print namelist
	#mfile = '/home/yangying/mq_data/topic_matric_origin.csv'
	#df = pd.read_csv(mfile)
	#df = df.ix[df['user_topic'].isin(namelist)]
	for q in querys[:]:
		for party in ['Republican Party','Democratic Party']:
		#for party in [1,4]:
			sdf = ndf.ix[ndf['Class'] == party]
			print sdf[[q,'Class']]


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'need a filename'
		sys.exit()
	filename = sys.argv[1]
	datafile = sys.argv[2]
	#get_querys(filename)
	#get_stat(filename)
	#best(filename)
	extract_data(filename,datafile)
