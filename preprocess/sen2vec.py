# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
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
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

Filemodel = config.Filemodel
Stoplist = config.Stoplist


def relevant_doc2vec(col1_name,col2_name):
	col1_name = col1_name.decode('gbk','ignore')
	col2_name = col2_name.decode('gbk','ignore')
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
		if len(keyword) == 0:
			return 0
		corpus.append(' '.join([t[0].replace('-',' ') for t in keyword])) #merge keyword
	countvect = CountVectorizer()
	#print corpus 

	m = countvect.fit_transform(corpus)
	m = m.toarray()
	sim = cosine_similarity(m,m)
	return sim[0][1]


def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs

model = gensim.models.KeyedVectors.load_word2vec_format('/home/yangying/is_that_a_duplicate_quora_question/data/GoogleNews-vectors-negative300.bin', binary=True)	

def sent2vec(s):

	words = str(s).lower().decode('utf-8','ignore')
	words = word_tokenize(words)
	words = [w for w in words if not w in stop_words]
	words = [w for w in words if w.isalpha()]
	#print words
	M = []
	for w in words:
		try:
			M.append(model[w])
		except:
			continue
	M = np.array(M)
	v = M.sum(axis=0)
	#print M,v
	if len(M) == 0 :
		return []
	else:
		return v / np.sqrt((v ** 2).sum())		

def get_corpus(filename):
	query_set = read_probs(filename)
	qt_set = []
	for i in range(len(query_set)-1):
		q = query_set[i]
		#print q[0][1]
		qt_set.append(q[0][1])
	#print qt_set
	return qt_set

def tf_idf():
	def preprocess(s):
		words = str(s).lower().decode('gbk')
		#print words
		words = word_tokenize(words)
		words = [w for w in words if not w in stop_words]
		words = [w for w in words if w.isalpha()]
		return words
			
	documents = get_corpus('c')	
	
	
	texts = [[word for word in preprocess(document)] for document in documents]
	#print texts
	dictionary = corpora.Dictionary(texts)
	
	corpus = [dictionary.doc2bow(text) for text in texts]
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	return corpus_tfidf

	
def compute_distance(filename):
	
	
	query_set = read_probs(filename)
	xname = ['i','j','vote_i','vote_j','pro_i','pro_j','sen_dis','sim_doc2vec' ,'sim_rake','tfdis']
	print xname
	corpus_tfidf = tf_idf()
	#for i in range(len(query_set)-1):
	for i in range(len(query_set)-1):
		q1 = query_set[i]
		
		if q1[1][2] > 10 :
			for j in range(i+1,len(query_set)-1):
				q2 = query_set[j]
				#print q1,q2
				if q2[1][2] > 10 :
					
					qt1 = q1[0][1]
					qt2 = q2[0][1]
					qi1 = q1[0][0]
					qi2 = q2[0][0]										
					qv1 = sent2vec(qt1)
					qv2 = sent2vec(qt2)
					#print qt1,qt2
					#print qv1,qv2
					if not (qv1 ==[] or qv2 ==[]):
						sen_dis = cosine(qv1,qv2)
						#print dis
						sim_doc2vec = relevant_doc2vec(qt1,qt2)
						sim_rake = relevant_rake(qt1,qt2)
						#print sim_doc2vec,sim_rake

						c = [corpus_tfidf[qi1],corpus_tfidf[qi2]]
						index = similarities.MatrixSimilarity(c)
						a = [ n.tolist() for n in index]
						tfdis = a[0][1]
						
						#print i,j,qi1,qi2,dis4
						
						print [q1[0][0],q2[0][0],q1[1],q2[1],sen_dis,sim_doc2vec ,sim_rake,tfdis]

def compute_tfidf(filename):
	
	
	query_set = read_probs(filename)
	xname = ['i','j','vote_i','vote_j','sen_dis','sim_doc2vec' ,'sim_rake','tfdis']
	print xname
	corpus_tfidf = tf_idf()
	#for i in range(len(query_set)-1):
	for i in range(len(query_set)-1):
		q1 = query_set[i]
		
		if q1[1][2] > 50 :
			for j in range(i+1,len(query_set)-1):
				q2 = query_set[j]
				#print q1,q2
				if q2[1][2] > 50 :
					
					qt1 = q1[0][1]
					qt2 = q2[0][1]
					qi1 = q1[0][0]
					qi2 = q2[0][0]					
			
					sim_rake = relevant_rake(qt1,qt2)
					#print sim_doc2vec,sim_rake

					c = [corpus_tfidf[qi1],corpus_tfidf[qi2]]
					index = similarities.MatrixSimilarity(c)
					a = [ n.tolist() for n in index]
					tfdis = a[0][1]
					
					#print i,j,qi1,qi2,dis4
					#if tfdis > 0 or sim_rake > 0:
						
					qv1 = sent2vec(qt1)
					qv2 = sent2vec(qt2)
					if not (qv1 ==[] or qv2 ==[]):
						sen_dis = cosine(qv1,qv2)
						sim_doc2vec = relevant_doc2vec(qt1,qt2)
						print [q1[0][0],q2[0][0],q1[1],q2[1],sen_dis,sim_doc2vec ,sim_rake,tfdis]
					
def main():
	#filename = '../mq_data/topic_matric_origin.csv'
	#filename = '../mq_result/test_m.csv'
	#pair_file = '../mq_result/stat/new_matrix/query_pair_l1'
	query_vpro = '../../mq_result/stat/new_matrix/query_vpro_all'
	#probs_file = '../mq_result/stat/stu.pro'
	#rule = {'Occupation':['Student']}
	
	probs_file = '../mq_result/stat/new_matrix/white_new.pro'
	#probs_file = '../mq_result/stat/white.pro'
	rule = {'Ethnicity':'White'}
	'''	
	if not os.path.exists(probs_file):
		merge.get_probs_file(rule,probs_file,False)	
	if not os.path.exists(pair_file):
		choose_query_pair(pair_file)
	'''
	#get_corpus(query_vpro)
	#tf_idf_all()
	#compute_distance(query_vpro)
	compute_tfidf(query_vpro)
			
	

if __name__ == '__main__':
	main()