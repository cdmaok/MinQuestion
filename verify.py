# -*- coding: utf-8 -*-
import pandas as pd 
import merge
import os
import numpy as np
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from knntext.doc2vec import getvector
from sklearn.feature_extraction.text import CountVectorizer
import RAKE
import config
import random
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import gensim
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

Filemodel = config.Filemodel
Stoplist = config.Stoplist


def new_metric_inter(col1,col2,prob):
	l = len(col1)
	miss2 = 0
	bingji = 0
	inter = 0
	single = 0
	equal = 0
	
	for i in range(l):
		if prob[i][1] == 1: #0401,加了用户群体条件
			if col1[i] == '?' and col2[i] == '?':
				miss2 = miss2 + 1
			else:
				bingji = bingji + 1
				if col1[i] == '?' or col2[i] == '?':
					single = single + 1
				else: 
					inter = inter + 1
					if col1[i] == col2[i]:
						equal = equal + 1
	
	
	if not bingji == 0:
		pro = inter / float(bingji)
		pro2 = equal / float(bingji)
	else: 
		pro = 0
		pro2 = 0
	
		
	return equal,inter,bingji,pro,pro2


def predictability2(col1,col2,prob):
	l = len(col1)
	miss2 = 0
	vote = 0
	inter = 0
	miss1 = 0
	for i in range(l):
		if prob[i][1] == 1: #0401,加了用户群体条件
			if col2[i] == '?':
				miss2 = miss2 + 1
			else:
				if not col1[i] == '?':# add this line 0328
					vote = vote + 1
					if col1[i] == col2[i]:
						inter = inter + 1
				else: miss1 = miss1 + 1
	
	group = miss2 + vote + miss1
	q2 = miss1 + vote
	
	if not vote == 0:
		pre2 = inter / float(vote)
	else: pre2 = 0
	
	if not q2 == 0:
		e = 1 / float(q2)
	else: e = 0 			
	
	den = vote + e * miss1
	if not den == 0:
		pre3 = inter / float(den)
	else: pre3= 0
		
	return inter,vote,q2,group,pre2,pre3
	
def predictability(col1,col2,prob):
	l = len(col1)
	miss2 = 0
	vote = 0
	inter = 0
	miss1 = 0
	for i in range(l):
		#print col2[i]
		if col2[i] == '?':
			miss2 = miss2 + 1
		else:
			if not col1[i] == '?':# add this line 0328
				vote = vote + 1
				if col1[i] == col2[i]:
					inter = inter + 1
			else: miss1 = miss1 + 1
	
	group = miss2 + vote + miss1
	q2 = miss1 + vote
	
	if not vote == 0:
		pre2 = inter / float(vote)
	else: pre2 = 0
				
	e = 1 / float(q2)
	den = vote + e * miss1
	if not den == 0:
		pre3 = inter / float(den)
	else: pre3= 0
	
	return inter,vote,q2,group,pre2,pre3
			

def relevant_doc2vec(col1_name,col2_name):
	vectors = [list(getvector(q,Filemodel)) for q in [col1_name,col2_name]]
	m = np.asarray(vectors)
	sim = cosine_similarity(m,m)
	return sim[0][1]

def relevant_rake(col1_name,col2_name):
	col1_name = col1_name.decode('gbk','ignore')
	col2_name = col2_name.decode('gbk','ignore')
	Rake = RAKE.Rake(Stoplist)
	corpus = []
	for q in [col1_name,col2_name]:
		keyword = Rake.run(q) ##return keyword
		corpus.append(' '.join([t[0].replace('-',' ') for t in keyword])) #merge keyword
	countvect = CountVectorizer()
	m = countvect.fit_transform(corpus)
	m = m.toarray()
	sim = cosine_similarity(m,m)
	return sim[0][1]

	
def response_bias_old(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	group = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 	
			if prob[i][1] == 1:
				group = group + 1
				if col1[i] == 'yes' or col1[i] == 'no':
					inter = inter + 1
				else: miss1 = miss1 + 1
	
	if not group == 0:
		res = inter / float(group)
		nres = miss1/ float(group)
	else: 
		res = 0
		nres=0
	#print inter,group,res
	return res,nres

def response_bias(col1,col2,prob):
	l = len(col1)
	group = 0
	vote = 0
	no_vote = 0
	bingji = 0
	miss = 0
	for i in range(l):
		if prob[i][1] == 1 or not col1[i] == '?':
			bingji = bingji + 1
		if prob[i][1] == 1:
			group = group + 1
			if not col1[i] == '?':
				vote = vote + 1
			else: no_vote = no_vote + 1
		else: miss = miss + 1
	
	if not group == 0:
		res = vote / float(group)
		nres = no_vote / float(group)
	else: 
		res = 0
		nres = 0

	if not bingji == 0:
		res2 = vote / float(bingji)
		nres2 = no_vote / float(bingji)
	else: 
		res2 = 0	
		nres2 = 0
		
	res3 = vote / float(l)
	nres3 = no_vote / float(l)
	return res,res2,res3,nres,nres2,nres3
	
def coverage_bias(col1,col2,prob):
	l = len(col1)
	vote = 0
	q1 = 0
	for i in range(l):
		if not col1[i] == '?':
			q1 = q1 + 1
			if prob[i][1] == 1:
				vote = vote + 1
	
	if not q1 == 0:
		cov = vote / float(q1)
	else: cov = 0
	return cov
	
def coverage_bias_old(col1,col2,prob):
	l = len(col1)
	miss1 = 0
	vote = 0
	inter = 0
	union = 0
	for i in range(l):
		if not col2[i] == '?': 
			if col1[i] == '?':
				miss1 = miss1 + 1
			else:
				vote = vote + 1
				if prob[i][1] == 1:
					inter = inter + 1
	if not vote == 0:
		cov = inter / float(vote)
	else: cov = 0
	#print inter,vote,cov
	return cov


def choose_query_pair(pair_file):
	qstat_file = '../mq_result/stat/count_comment' 
	qstat = [eval(e.strip()) for e in open(qstat_file).readlines() ]
	c1 = 0
	c2 = 0
	candi_list = []
	goal_list = []
	for q in qstat:
		if q[3] >= 5 and q[3] < 100:
			print q
			candi_list.append(q[0])
			c1= c1+1
		elif q[3] >= 50:
			print q
			goal_list.append(q[0])
			c2= c2+1
	print c1,c2
	print candi_list
	print goal_list	
	f = open(pair_file,'a')
	f.write(str(candi_list)+'\n')
	f.write(str(goal_list)+'\n')
	f.close()
	

#def preprocess():


def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs
	
def compute_metric(filename,pair_file,probs_file):
	
	xname = ['i','j','inter','vote','q2','group','pre2','pre3','_inter','_vote','_q2','_group','_pre2','_pre3','sim_doc2vec','sim_rake','response','response2','response3','nonresponse','nonresponse2','nonresponse3','coverage']
	print xname
	pair_ij = read_probs(pair_file)	
	prob = read_probs(probs_file)
	#list = pair_ij[0] + pair_ij[1]
	df = pd.read_csv(filename,dtype={"user_topic":str,"Class":str})		
	#list = []
	for j in pair_ij[1]:
		for i in pair_ij[0]:
			if not i == j : 
				col1 = df.ix[:,i]
				col2 = df.ix[:,j]

				col1_name = df.iloc[:,i].name
				col2_name = df.iloc[:,j].name

				inter,vote,q2,group,pre2,pre3 = predictability(col1,col2,prob)
				_inter,_vote,_q2,_group,_pre2,_pre3 = predictability2(col1,col2,prob)
				# print col1_name,col2_name
				sim_doc2vec = relevant_doc2vec(col1_name,col2_name)
				sim_rake = relevant_rake(col1_name,col2_name)
				
				res,res2,res3,nres,nres2,nres3 = response_bias(col1,col2,prob)
				cov = coverage_bias(col1,col2,prob)
				'''
				# print rel_doc2vec
				if rel_rake>0:
					print col1_name,col2_name
					print rel_rake
				'''	
				#print [i,j,pre,rel_doc2vec,rel_rake,pres,res,nres,cov,gpres,gres,gnres,gcov]
				print [i,j,inter,vote,q2,group,pre2,pre3,_inter,_vote,_q2,_group,_pre2,_pre3,sim_doc2vec,sim_rake,res,res2,res3,nres,nres2,nres3,cov]	

def random_compute_metric(filename,pair_file,probs_file):
	
	
	#xname = ['i', 'j', 'votei', 'votej', 'sim_sen2vec', 'sim_doc2vec', 'sim_rake', 'sim_tfidf', 'vote_cosine', 'vote_manhattan', 'vote_euclidean','equal','inter','bingji','pro','pro2','_inter','_vote','_q2','_group','_pre2','_pre3','res','res2','res3','nres','nres2','nres3','cov']
	xname = ['i', 'j', 'votei', 'votej', 'sim_sen2vec', 'sim_doc2vec', 'sim_rake', 'sim_tfidf', 'vote_cosine', 'vote_manhattan', 'vote_euclidean','equal','inter','bingji','pro','pro2','inter','vote','q2','group','pre2','pre3','_res','_nres','cov']
	print xname	
	pair_ij = read_probs(pair_file)	
	prob = read_probs(probs_file)
	df = pd.read_csv(filename,index_col=0,dtype={"user_topic":str,"Class":str})

	for i in range(1,len(pair_ij)):
		#print p
		p = pair_ij[i]
		
		vote_i = p[2]
		vote_j = p[3]
		if vote_i > vote_j:	# >:more_less  <: less_more			
			i = p[0]
			j = p[1]
		else:
			i = p[1]
			j = p[0]
		
		col1 = df.ix[:,i]
		col2 = df.ix[:,j]
		#print i,j
		equal,inter,bingji,pro,pro2 = new_metric_inter(col1,col2,prob)
		
		
		inter,vote,q2,group,pre2,pre3 = predictability(col1,col2,prob)
		_res,_nres = response_bias_old(col1,col2,prob)
		cov = coverage_bias_old(col1,col2,prob)
		print p + [equal,inter,bingji,pro,pro2,inter,vote,q2,group,pre2,pre3,_res,_nres,cov]
		'''
		_inter,_vote,_q2,_group,_pre2,_pre3 = predictability2(col1,col2,prob)		
		res,res2,res3,nres,nres2,nres3 = response_bias(col1,col2,prob)
		cov = coverage_bias(col1,col2,prob)
		
		print p + [equal,inter,bingji,pro,pro2,_inter,_vote,_q2,_group,_pre2,_pre3,res,res2,res3,nres,nres2,nres3,cov]
		'''
		

		
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/yangying/is_that_a_duplicate_quora_question/data/GoogleNews-vectors-negative300.bin', binary=True)	
	
def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())		
						
def compute_distance(filename):
	
	query_set = read_probs(filename)
	xname = ['i','j','vote_i','vote_j','dis_cosine','dis2','dis3']
	print xname	
	#for i in range(len(query_set)-1):
	for i in range(len(query_set)-1):
		q1 = query_set[i]
		if q1[1][2] > 0 :
			for j in range(i+1,len(query_set)-1):
				q2 = query_set[j]
				if q2[1][2] > 0 :
					v1 = numpy.array([q1[1][4],q1[1][5]])
					v2 = numpy.array([q2[1][4],q2[1][5]])
					#print v1,v2
					dis1 = 1 - cosine(v1,v2)
					if dis1 != dis1:
						dis1 = 0
					dis2 = abs(q1[1][4] - q2[1][4])
					dis3 = (q1[1][4] - q2[1][4])**2
					
					#qt1 = q1[0][1]
					#qt2 = q2[0][1]
					#qt1 = sent2vec(q1[0][1])
					#qt2 = sent2vec(q2[0][1])
					#print qt1,qt2
					#sim_doc2vec = relevant_doc2vec(qt1,qt2)
					#sim_rake = relevant_rake(qt1,qt2)
					print [i,j,q1[1][2],q2[1][2],dis1,dis2,dis3]
					
	
					
def print_csv(filename,pair_file,probs_file):
	
	df = pd.read_csv(filename,dtype={"user_topic":str,"Class":str})
	
	pair_ij = read_probs(pair_file)	
	prob = read_probs(probs_file)

	#query_list = [0] + pair_ij[0] + pair_ij[1]
	query_list = [0] + pair_ij[1]
	#print query_list
	tmp_df = df.ix[:,query_list]
	
	tmp_df.to_csv('../mq_result/stat/query_pair_100.csv',index = False)

def processdf(filename):
	
	df = pd.read_csv(filename,index_col=0,dtype={"user_topic":str,"Class":str})
	df = df.replace('yes',1).replace('no',-1).replace('?',numpy.nan)
	#df = df.replace('yes',1).replace('no',-1).replace('?',0)
	df.to_csv('../mq_result/test_m_m11.csv')
	
	#return df	
				
def main():
	filename = '../mq_data/topic_matric_origin.csv'
	#filename = '../mq_result/test_m.csv'
	pair_file = '../mq_result/stat/new_matrix/0408/10_white_sim_dis.log'
	#query_vpro = '../mq_result/stat/new_matrix/query_vpro_stu'
	#probs_file = '../mq_result/stat/stu.pro'
	#rule = {'Occupation':['Student']}
	
	#probs_file = '../mq_result/stat/new_matrix/white_new.pro'
	probs_file = '../mq_result/stat/white.pro'
	rule = {'Ethnicity':'White'}
		
	if not os.path.exists(probs_file):
		merge.get_probs_file(rule,probs_file,False)	
	if not os.path.exists(pair_file):
		choose_query_pair(pair_file)
	'''	
	df = pd.read_csv(filename,dtype={"user_topic":str,"Class":str})
	querylists = df.columns.values.tolist()
	col1_name= 'Should marijuana be illegal?  '
	col2_name= 'Should marijuana be decriminalized?'
	index = querylists.index(col1_name)
	print index
	#print df[col1_name]
	'''
	#processdf(filename)
	#print_csv(filename,pair_file,probs_file)
	#print_pre_query(filename,pair_file,probs_file)
	random_compute_metric(filename,pair_file,probs_file)
	#sim = relevant_doc2vec(col1_name,col2_name)
	#print sim
	#compute_metric(filename,pair_file,probs_file)
	#print_sim_query(filename,pair_file,probs_file)
	#compute_distance(query_vpro)
			
	

if __name__ == '__main__':
	main()