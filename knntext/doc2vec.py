#coding=utf-8

### try to combine the user query matrix

import matplotlib
matplotlib.use('Agg')  #,图形并没有在屏幕上显示,但是已保存到文件,
import numpy as np
from nltk import word_tokenize
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec






def getSen(content):
	return [ LabeledSentence(words=word_tokenize(line),tags=[str(row)])    for row,line in enumerate(content)]

def train(content,modelfile):
	sentences = []
	sentences += getSen(content)

	global model
	model = Doc2Vec(alpha=0.025,min_alpha=0.025)
	model.build_vocab(sentences)
	for epoch in range(10):
		model.train(sentences)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
	model.save(modelfile)
	# print model.most_similar(u'government')

model = None

def getMatrix(matrixfile,modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	np.savetxt(matrixfile,model.docvecs.doctag_syn0)
	return model.docvecs.doctag_syn0

def getvector(string,modelfile):
	global model
	if model == None:
		model = Doc2Vec.load(modelfile)
	return model.infer_vector(word_tokenize(string))


# if __name__ == '__main__':
# 	df = pd.read_csv('./user_topic_origin.csv')
# 	querylist = df.columns.values[1:-1]  #第一行
# 	querylist = list(querylist)
#
# 	for iter in xrange(1):
# 		username = group(iter)  ##同为0组的用户
# 		content = context(querylist,username)
# 		filemodel = './doc2vec.model'
# 		train(content,filemodel)
		# x = 'Is nationalism a good thing?'
		# y = 'Will World War III happen?'
		# xv = getvetor(x,filemodel)
		# yv = getvetor(y,filemodel)
		# print xv
		# print yv
		# print cosine_similarity(xv,yv)
		# m = getMatrix('doc2vec.matrix',filemodel)
        #
		# sim = cosine_similarity(m,m) #计算相似性
		# edge = analy(sim,querylist)
		# for i in xrange(len(edge)):
		# 	print querylist[edge[i][0]]
		# 	print querylist[edge[i][1]]
		# 	print '\n'

