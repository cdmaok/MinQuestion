#coding=utf-8
import numpy as np
import numpy
import pandas as pd
import sys
sys.path.append("..")
import config
import merge
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
data_path = config.data_path
result_path = config.result_path


def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs

def find(dis_file):
	dict = {}
	dis = [eval(e.strip()) for e in open(dis_file).readlines() ]

	for i in range(1,len(dis)-1):
		e = dis[i]
		#print e
		dict[e[0][0]] = [e]
		
		'''
		if e[0][0] in dict:
			dict[e[0][0]].append(i)			
		else:
			dict[e[0][0]] = [i]
		'''
	#print dict
	'''
	if key in dict:
		for j in dict[key]:
			e = dis[j]
			if e[1] == key2:			
				#print e
				tmp = e
			#else: print 'no'
	'''
	return dis,dict

def judge_polarity(q1,q2):
	return True
	
	
def get_query_intersection(userfile,dis_file):
	
	all_pair = read_probs(userfile)
	dis,dict = find(dis_file)
	xname = ['i', 'j', 'votei', 'votej', 'sim_sen2vec', 'sim_doc2vec', 'sim_rake', 'sim_tfidf', 'vote_cosine', 'vote_manhattan', 'vote_euclidean']
	print xname
	for i in range(1,len(all_pair)):
		
		a = all_pair[i]		
		key = a[0]
		key2 = a[1]
		q1 = dict[key][0]
		q2 = dict[key2][0]
		
		v1 = numpy.array([q1[1][4],q1[1][5]])
		flag = judge_polarity(q1,q2)
		if flag == True: # v1 yes- v2 yes  v1 no- v2 no
			v2 = numpy.array([q2[1][4],q2[1][5]])
		else: # v1 yes - v2 no  v1 no - v2 yes
			v2 = numpy.array([q2[1][5],q2[1][4]])
		#print v1,v2
		dis1 = 1-cosine(v1,v2)
		if dis1 != dis1:
			dis1 = 0
		dis2 = abs(q1[1][4] - q2[1][4])
		dis3 = (q1[1][4] - q2[1][4])**2
		
		print [key,key2,q1[1][2],q2[1][2],a[4],a[5],a[6],a[7],dis1,dis2,dis3]
		
		
		
		
	
def main():
	userfile = result_path + 'stat/new_matrix/0408/all50_sim'
	goal_pro = result_path + 'stat/new_matrix/query_vpro_white'
	#userfile = result_path + 'stat/new_matrix/find.txt'
	#goal_pro = result_path + 'stat/new_matrix/test.txt'
	get_query_intersection(userfile,goal_pro)
	#get_user_pro(goal_pro)

			
if __name__ == '__main__':
	main()