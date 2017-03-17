#coding=utf-8

### this script is to analyse the exp data
import config
import sys
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
import collections
## demo -->4 repubG --> 1

def get_missing(filename):
	df = pd.read_csv(filename)
        query = df.columns.values[:]
        origin_matrix = df.as_matrix()
	positions = []
	print origin_matrix.shape
        for iter_i,i in enumerate(origin_matrix):
                for iter_j,j in enumerate(i):
                        if j=='yes' or j=='no':
                                positions.append((iter_i,iter_j))
	print positions
	return positions



def box_plot_matrix():
	fs = ['../mq_result/white_old_biscaler0.6twoparty_balan_reduce.csv','../mq_result/white_old_knn0.6twoparty_balan_reduce.csv','../mq_result/white_old_knn_doc0.6twoparty_balan_reduce.csv','../mq_result/white_old_knn_rake0.6twoparty_balan_reduce.csv','../mq_result/white_old_itersvd0.6twoparty_balan_reduce.csv','../mq_result/white_old_mf0.6twoparty_balan_reduce.csv','../mq_result/white_old_knn_simrank0.6twoparty_balan_reduce.csv','../mq_result/white_old_soft_impute0.6twoparty_balan_reduce.csv']
	pos = get_missing('../mq_data/topic_matric_twoparty_balan.csv')
	ms = [get_matrix(f,pos) for f in fs[:]]
	fig = plt.figure()
	plt.boxplot(ms,sym='')
	old = range(1,9)
	print old
	new = ['bs','knn_text','knn_doc','knn_rake','iter-svd','mf','knn_sim','Impute']
	plt.xticks(old[:],new[:])
	fig.savefig('test.png')




def get_matrix(filename,pos):
	df = pd.read_csv(filename)
	#columns = df.columns.values.tolist()
	#columns.remove('user_topic')
	#columns.remove('Class')
	#df = df[columns]
	m = df.as_matrix()
	res = []
	for p in pos:
		res.append(m[p[0]][p[1]])
	print res
	return res




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

def get_stat(filename,row=2,col=0):
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

def newdf(datafile):
	df = pd.read_csv(datafile)
	columns = df.columns.values.tolist()
	columns = [col.strip() for col in columns]
	ndf = pd.DataFrame(data=df.as_matrix(),columns=columns)
	return ndf

def extract_data(logfile,datafile=None):
	datafile = sys.argv[2]
	if datafile == None:
		print 'need datafile name'
		sys.exit()
	querys = best(logfile).split('\n')
	### Class
	df = newdf(datafile)
	namelist = df['user_topic'].tolist()
	print df['Class']
	for q in querys[:]:
		#for party in ['Republican Party','Democratic Party']:
		print q
		for party in [1,4]:
			if party == 1:
				print 'repub'
			else:
				print 'de'
			print len(namelist)
			getVector(df,party,q)

def extract_qn(logfile,datafile=None):
	datafile = sys.argv[2]
	if datafile == None:
		print 'need datafile name'
		sys.exit()
	querys = best(logfile).split('\n')
	draw_graph(querys,datafile)

def draw_graph(querys,datafile):
	df = newdf(datafile)
	x = df[querys].as_matrix()
	y = list(df['Class'].as_matrix())
	dt = DecisionTreeClassifier(criterion='gini',min_samples_split=15)
	x[x>0] = 1
	x[x<0] = -1
	print x.shape,len(y)
	dt.fit(x,y)
	classlist = ['Republican Party','Democratic Party']
        dot = export_graphviz(dt,filled=True,label='all',feature_names=querys,leaves_parallel=False,class_names=classlist,out_file=None)
        graph = pydotplus.graph_from_dot_data(dot)
        graph.write_pdf('./tt.pdf')


def getVector(df,party,q):
	sdf = df.ix[df['Class'] == party]
	vs = sdf[q].as_matrix()
	total = vs.shape[0]
	neg = (vs > 0).sum()
	print total,neg,total - neg
	return (neg,total - neg)


def get_full_rank(rank_file,start,size = 10):
	i = 0
	rcontents = open(rank_file).readlines()
	startIndex = start * 11 + 1
	endIndex = startIndex + 10
	ranks = [eval(string)  for string in rcontents[startIndex:endIndex]]
	votelist = []
	for rank in ranks:
		votelist += rank[:size]
	votedict = collections.Counter(votelist)
	topquery = [ e[0] for e in votedict.most_common(size)]
	return topquery

		


def get_full_topic(rank_file,log_file,data_file,row = 0,col = 0):
	print 'need parameter: rank_file,log_file,data_file'
	df = pd.read_csv(data_file)
	columns = df.columns.values.tolist()
	columns.remove('user_topic')
	columns.remove('Class')
	querys = columns
	precs = get_stat(log_file,row = row,col = col)
	index = sorted(range(len(precs)),key = lambda x: precs[x],reverse = True)
	start = index[0]
	voterank = get_full_rank(rank_file,start)
	votequery = [querys[i].strip() for i in voterank]
	print '\n'.join(votequery)
	draw_graph(votequery,data_file)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'need a filename'
		sys.exit()
	filename = sys.argv[1]
	#get_querys(filename)
	#get_stat(filename)
	#print best(filename)
	#extract_data(filename)
	#extract_qn(filename)
	#box_plot_matrix()
	get_full_topic(sys.argv[1],sys.argv[2],sys.argv[3])
