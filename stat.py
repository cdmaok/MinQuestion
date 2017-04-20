#coding=utf-8
import pandas as pd 
import config
from collections import Counter

def sum_comment(qstat_file):
	#qstat_file = '../mq_result/stat/new_matrix/count_comment' 
	data = open(qstat_file)
	next(data)
	qstat = [eval(e.strip()) for e in data]
	#print qstat
	vote = []
	for q in qstat:
		#print q
		vote.append(q[9])
	
	for i in Counter(vote).most_common():
		a = [i[0],i[1]]
		print(a)		
		
		f = open('../mq_result/stat/new_matrix/sum_query_intersection','a')
		f.write(str(a)+'\n')
		f.close

def sum_vote(qstat_file):
	#qstat_file = '../mq_result/stat/new_matrix/count_comment' 
	data = open(qstat_file)
	next(data)
	qstat = [eval(e.strip()) for e in data]
	#print qstat
	votei = [str(q[2])+'_'+str(q[3]) for q in qstat]
	#print votei
	#votej = [[q[1],q[3]] for q in qstat]
	
	for i in Counter(votei).most_common():
		a = [i[0],i[1]]
		print(a)		
		
		f = open('../mq_result/stat/new_matrix/sum_stu_vote','a')
		f.write(str(a)+'\n')
		f.close	
	
def read(query_file):
	qstat = [eval(e.strip()) for e in open(query_file).readlines() ]
		
	for q in qstat:
		comment = q[0][1] + ' '
		#comment = ''
		for u in range(1,len(q)):
			if not q[u][0] == '':
				comment = comment+ q[u][2]
		print comment
		 
def count(query_file):
	#file_name="hello.txt"

	line_counts=0
	word_counts=0
	char_counts=0

	file=open(query_file,"r")
	for line in file.readlines():
		words=line.split(' ')
		line_counts+=1
		word_counts+=len(words)
		char_counts+=len(line)
		 
	print "line_count",line_counts
	print "word_count",word_counts
	print "char_count",char_counts	
		
def choose_query_pair(qstat_file):
	#qstat_file = '../mq_result/stat/new_matrix/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	vote = []
	
	for q in qstat:
		#print q
		print [q[0],q[1],q[3],q[4]]
	#print vote		

def process_gt(filename):
	qstat = [e for e in open(filename).readlines() ]
	query_file = '../mq_result/stat/new_matrix/query_vpro_all'
	pro = [eval(e.strip()) for e in open(query_file).readlines() ]
	
	for q in qstat:
		a = q.split('[')
		f = []
		for i in a :			
			j = i.split(' ')
			#print j[0]
			if not j[0] == '':
				f.append(pro[int(j[0])][0])
		print f
	
	
	
def filter_query(qstat_file):
	#qstat_file = '../mq_result/stat/new_matrix/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	#c = 0
	for q in qstat:
		if q[2] > 1:
			print q
			f = open('../mq_result/stat/new_matrix/query_intersection_l1','a')
			f.write(str(q)+'\n')
			f.close
			#c= c+1
	#print c

	
### count the comment of each query
def count_query_comment(filename):
	df = pd.read_csv(filename)
	print df.shape
	query = df.columns.values[:]
	#print query[0]
	origin_matrix = df.as_matrix()
	qnum = origin_matrix.shape[1]
	#print df.ix[:,1]
	'''
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
		f = open('../mq_result/stat/new_matrix/count_comment','a')
		f.write(str(list)+'\n')
		f.close
		
		#print q,yes,no,total
	'''
	
	
if __name__ == '__main__':
	filename = '../mq_result/stat/new_matrix/data/gt_no.txt'
	process_gt(filename)
	#filename = '../mq_data/topic_matric_origin.csv'
	filename = '../mq_result/test_m.csv'
	#stat_file = '../mq_result/stat/new_matrix/query_intersection_l1'
	query_file = '../mq_result/stat/new_matrix/data/query_user_vote_comment'
	#read(query_file)
	#count_query_comment(filename)
	stat_file = '../mq_result/stat/new_matrix/less_more.log' 
	stat_file = '../mq_result/stat/new_matrix/dis_stu'
	#sum_vote(stat_file)
	#count(query_file)
	#filter_query(stat_file)
	#choose_query_pair(query_file)
	
	
