#coding=utf-8
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import config
import merge
data_path = config.data_path
result_path = config.result_path

def get_sparse_martix(userfile,goalfile):	
	df = pd.read_csv(userfile,index_col=0,dtype={"user_topic":str,"Class":str})
	newdf = df.replace('?',np.nan)
	#x = newdf[columns].as_matrix()
	len = newdf.shape[1]-1
	print len
	query = []
	for i in range(len):
		tem_col = newdf.iloc[:,i]
		#for k1, group in tem_col.groupby(tem_col):
		#print tem_col.groupby(tem_col).size()
		sum = 0
		for j in tem_col.groupby(tem_col).size():
			sum = sum + j
		if(sum > 1):
			query.append(i)
			print i,sum
			#print "------group-----",k1,group

	
	bdf = newdf.iloc[:,query]
	bdf = bdf.dropna(how = 'all')
	users = list(bdf.index.values)
	print users
	query = query + [-1]
	
	final_df = df.ix[users,query]
	final_df.to_csv(goalfile)
	
def get_query_user(userfile):	
	df = pd.read_csv(userfile,index_col=0,dtype={"user_topic":str,"Class":str})
	newdf = df.replace('?',np.nan)
	#x = newdf[columns].as_matrix()
	len = newdf.shape[1]-1
	#print len
	query = []
	for i in range(len):
		#newdfdf.iloc[:,i].name
		tem_col = newdf.iloc[:,i]
		tem_col = tem_col.dropna(how = 'all')
		print tem_col.index.values.tolist()

def group(filename):
	query_file = filename
	dict = {}
	file=open(query_file,"r")
	for line in file.readlines():
		w = line.strip().split(',')
		dict[w[0]] = '0'
		#if not w[1] == '':
			#dict[w[0]] = w[1]
		if not w == '':
			dict[w[0]] = w
	#print dict
	return dict
	
		
def get_query_user_vote(userfile):	
	df = pd.read_csv(userfile,index_col=0,dtype={"user_topic":str,"Class":str})
	newdf = df.replace('?',np.nan)
	#x = newdf[columns].as_matrix()
	length = newdf.shape[1]-1
	#print len
	query = []
	#dict = group('./group2.csv')
	dict = group('./data/fulldata_xmeans.csv')
	#dict2 = group('./age.csv')
	for i in range(length):
		#newdfdf.iloc[:,i].name		
		tem_col = newdf.iloc[:,i]
		tem_col = tem_col.dropna(how = 'all')
		ind = tem_col.index.values.tolist()
		#vote = tem_col.tolist()
		info = [[i,tem_col.name]]
		for j in ind:
			info = info + [[j,tem_col.ix[j,:],dict[j]]]
		#l = len(info)
		print info		
		
def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs		
		
def get_query_intersection(userfile):
	
	all_query = read_probs(userfile)

	for i in range(len(all_query)-1):
		for j in range(i+1,len(all_query)):
			#print all_query[i],type(all_query[i])
			a = all_query[i]
			b = all_query[j]
			c = list(set(a).intersection(set(b)))
			#print c,len(c)
			if len(c) > 0 :
				print [i,j,len(c),len(a),len(b),c]

def get_user_pro(pro_file):
	dict = {}
	probs = [eval(e.strip()) for e in open(pro_file).readlines() ]
	for e in probs:
		#print e,e[0],e[1]
		dict[e[0]] = e[1]	
	#print dict
	return dict
				
def get_group_query_votepro(userfile,pro_file):
	
	all_query = read_probs(userfile)
	all_pro = get_user_pro(pro_file)		
	group = 0
	for i in all_query:
		#print i
		yes = 0
		no = 0
		vote = 0
		y_pro = 0
		n_pro = 0
		other = 0
		for j in range(1,len(i)):
			#print j
			j_pro = 1
			j_pro = all_pro[i[j][0]]
			#print j,j_pro
			if j_pro == 1:
				group = group + 1
				if i[j][1] == 'yes':
					yes = yes + 1
				elif i[j][1] == 'no':
					no = no + 1
				vote = yes + no
				if not vote == 0 :
					y_pro = yes / float(vote)
					n_pro = no / float(vote)
				else: 
					y_pro = 0
					n_pro = 0
			else: other = other + 1
		total = vote+other
		if not total == (len(i)-1):
			print 'Error!!!'
			break
		print [i[0],[yes,no,vote,total,y_pro,n_pro]]
	print group
	
def get_sparse_pro(filename,spfilename,goal_pro):
	df = pd.read_csv(spfilename,index_col=0,dtype={"user_topic":str,"Class":str})
	sp_list = df.index.values.tolist()
	#print sp_list
	all_pro = read_probs(filename)
	pro_list = [i[0] for i in all_pro]
	#print pro_list
	print len(sp_list),len(pro_list)
	'''
	for i in sp_list:
		index = pro_list.index(i)
		print i,pro_list[index]
	'''
	tmp = [ (i,all_pro[pro_list.index(i)][1]) for i in sp_list]
	#print tmp
	merge.log_probs(tmp,goal_pro)
	
def main():
	userfile = data_path + 'topic_matric_origin.csv'
	#goalfile = result_path + 'test_m.csv'
	#pro_file = result_path + 'stat/stu.pro'
	#goal_pro = result_path + 'stat/new_matrix/white_new.pro'
	goal_pro = result_path + 'stat/stu.pro'
	query_user_list = result_path + 'stat/new_matrix/query_user_list'
	query_user_vote = result_path + 'stat/new_matrix/query_user_vote'
	#userfile = data_path + 'test_m.csv'
	#userfile = result_path + 'test_m.csv'
	#get_sparse_martix(userfile,goalfile)
	#get_sparse_pro(pro_file,goalfile,goal_pro)	
	#get_query_user(goalfile)
	#get_query_intersection(query_user_list)
	get_query_user_vote(userfile)
	#get_group_query_votepro(query_user_vote,goal_pro)
	#get_user_pro(goal_pro)
	#group()

			
if __name__ == '__main__':
	main()