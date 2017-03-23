#coding=utf-8
import pandas as pd 
import config
from collections import Counter

def sum_comment():
	qstat_file = '../mq_result/stat/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	vote = []
	for q in qstat:
		vote.append(q[3])
	for i in Counter(vote).most_common():
		a = [i[0],i[1]]
		print(a)		
		f = open('../mq_result/stat/sum_comment','a')
		f.write(str(a)+'\n')
		f.close
	
	
def filter_query():
	qstat_file = '../mq_result/stat/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	c = 0
	for q in qstat:
		if q[3] > 100:
			print q
			c= c+1
	print c

	
### count the comment of each query
def count_query_comment(filename):
	df = pd.read_csv(filename)
	query = df.columns.values[:]
	#print query[0]
	origin_matrix = df.as_matrix()
	qnum = origin_matrix.shape[1]
	#print df.ix[:,1]
	
	for q in range(1,qnum-1):
		#print q
		yes = 0
		no = 0
		vote = 0
		miss =0
		total = 0
		list = []
		for v in df.ix[:,q]:
			#print v
			if v == 'yes':
				yes = yes+1
			elif v == 'no':
				no = no+1
			else: miss=miss+1
			vote = yes + no
			total = vote + miss
		#print q,yes,no,vote,miss,total,query[q]
		list =[q,yes,no,vote,miss,total,query[q]]
		print list
		f = open('../mq_result/stat/count_comment','a')
		f.write(str(list)+'\n')
		f.close
		
		#print q,yes,no,total

	
	
if __name__ == '__main__':
	filename = '../mq_data/topic_matric_origin.csv'
	#count_query_comment(filename)
	#sum_comment()
	filter_query()
	
	
	
