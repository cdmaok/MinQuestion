#coding=utf-8
### research about doc2vec

from gensim.models.doc2vec import *
from nltk import word_tokenize
import os
import json
import numpy as np
import sklearn.metrics.pairwise as pw
import pandas as pd
from sklearn.cluster import AffinityPropagation,KMeans
import collections
from scipy.cluster.hierarchy import ward, dendrogram
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin

def getVetor(string,modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	return model.infer_vector(word_tokenize(string.decode('gbk')))
	
def tf_idf(querylist):				
	corpus = querylist
	#print len(corpus) 	
	#corpus = [[word for word in document] for document in documents]
	vectorizer=CountVectorizer()
	transformer=TfidfTransformer()
	tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
	word=vectorizer.get_feature_names()
	weight=tfidf.toarray()
	
	#print tfidf,type(tfidf)
	return weight
	
	
def read_probs(filename):
	probs = [e.strip() for e in open(filename).readlines() ]
	return probs	

def read_probs2(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs	

	
def get_query(filename):
	query_set = read_probs2(filename)
	qt_set = []
	for i in range(len(query_set)-1):
		q = query_set[i]
		qt_set.append(q[0][1])
	return qt_set	

def get_corpus(filename):
	query_set = read_probs(filename)
	#print query_set
	qt_set = []
	for i in range(len(query_set)):
		
		if len(query_set[i]) == 0 :
			print i
			print query_set[i]

def lda(querylist):			
	
	texts_filtered = querylist
	#print texts
	texts_stemmed = [[word for word in document.lower().split()] for document in texts_filtered]
	
	#texts_stemmed = [[docment] for docment in texts_filtered]
	all_stems = sum(texts_stemmed, [])
	stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
	texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
	
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	'''
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
	lsi_topics = lsi.print_topics(num_topics=100,num_words=20)
	for i in lsi_topics:
		print i
	'''
	
	lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
	lda_topics = lda.print_topics(num_topics=100,num_words=20)
	for i in lda_topics:
		print i
	
	
	

def AP(matrix):

	S = -euclidean_distances(matrix, squared=True)
	
	preference = np.median(S) * 3.39
	'''
	inf = float("inf")
	af = AffinityPropagation(preference=preference, affinity="precomputed")
	vector_precomputed = af.fit_predict(S)
	'''
	af = AffinityPropagation(preference=preference, verbose=True,max_iter=1000)
	#af = AffinityPropagation()
	vector = af.fit_predict(matrix)
	af.cluster_centers_indices_
	return vector		
	
def kmeans(matrix):
	
	clf = KMeans(n_clusters=121)
	vector = clf.fit_predict(matrix)
	
	return vector
	
def Querylist(modelfile,filename):

	queryFile = '../../mq_result/stat/new_matrix/query_vpro_all'
	querylist_ori = get_query(queryFile)

	querylist = read_probs(filename)
	#print querylist
	
	#v = [list(getVetor(q,modelfile)) for q in querylist]	
	#matrix = np.asarray(v)
	matrix = tf_idf(querylist)
	#lda(querylist)
	
	
	print matrix.shape
	
	#vector = kmeans(matrix)
	vector = AP(matrix)
	
	vector = [ str(v) for v in list(vector)]
	dict = {}
	
	for i in range(len(vector)):
		c = vector[i]
		if c in dict:
			dict[c].append(i)
		else:
			dict[c] = [i]
	list_d = []
	list_d = sorted(dict.keys(), key=lambda k: len(dict[k]),reverse=True) 	
	print '#cluster = ',len(list_d)
	for j in list_d:
		for q in dict[j]:
			print [j,q,querylist_ori[q]]
	

	

	
	
if __name__ == '__main__':
	#directory = './topic_comment'
	filemodel = '/home/yangying/data/doc2vec.model'
	#filename = '../../mq_result/stat/corpus_ps'
	filename = '../../mq_result/stat/new_matrix/data/text/title_comment_rem_stem'
	Querylist(filemodel,filename)
	#get_corpus(filename)
