#coding=utf-8
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fill.doc2vec import getvector
from sklearn.feature_extraction.text import CountVectorizer
import RAKE
import os
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert(0,'..')
import config
'''TODO: fill with matrix factorization change Vote Matrix'''

Vote_Matrix = config.Vote_Matrix
Filemodel = config.Filemodel
Stoplist = config.Stoplist
Result_path = '/home/yangying/MinQuestion/'

if __name__ == '__main__':
	df = pd.read_csv(Vote_Matrix)
	querylist = df.columns.values.tolist()  #第一�?	querylist.remove('user_topic')  #第一�?	querylist.remove('Class')
	querylist = [q.strip().decode('utf-8','ignore')  for q in list(querylist)]

	method = 'rake'
	if method == 'rake':
		corpus = []
		Rake = RAKE.Rake(Stoplist)
		for i,q in enumerate(querylist):
			keyword = Rake.run(q)
			corpus.append(' '.join([t[0].replace('-',' ') for t in keyword]))
		countvect = CountVectorizer()
		m = countvect.fit_transform(corpus)
		m = m.toarray()
	elif method =='doc2vec':
		vectors = [list(getvector(q,Filemodel)) for q in querylist]
		m = np.asarray(vectors)

	sim = cosine_similarity(m,m) #计算相似�?	ave = np.sum(sim,axis=1) / len(querylist)

	csvfile = file(Result_path +  method + '_query_similarity.csv','wb')
	writer = csv.writer(csvfile)
	writer.writerow(querylist)
	for i in xrange(len(querylist)):
		data = sim[i].tolist()
		data.append(ave[i])
		writer.writerow(data)

