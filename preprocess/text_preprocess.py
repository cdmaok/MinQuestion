# -*- coding: utf-8 -*-
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import os
import sys
sys.path.append("..")
import config
import json
Comment_Dir = config.Comment_Dir
data_path = config.data_path
import pandas as pd
from nltk.stem.porter import PorterStemmer
import nltk
from collections import Counter

def context(querylist,username):
    directory = Comment_Dir
    data = [[] for i in xrange(len(querylist))]
    for filename in os.listdir(directory):
        filepath = directory + '/' + filename
        #filepath = '../../mq_result/stat/new_matrix/data/comment_test'
        lines = [l.strip().decode('utf-8','ignore') for l in open(filepath).readlines()]
        for l in lines:
            l = l.replace('\\','/')
            dic = json.loads(l,strict= False)
            title = dic[u'title'].strip().encode('utf-8','ignore')
            text = dic[u'text'].strip().encode('utf-8','ignore')
            user = dic[u'name'].strip().encode('utf-8','ignore')
            point = dic[u'point'].strip().encode('utf-8','ignore')
            if title in querylist :
                #print 'yes'
                #print '----',l,title,text,user,point				
                ind = querylist.index(title)
                if data[ind] == []:
                    data[ind] = [[ind,title]]
                data[ind].append([user,point,text])
    for i in xrange(len(querylist)):
        if data[i] == []:
            data[i] = [[i,querylist[i].encode('utf-8','ignore')]]
        print data[i]
			
				
			
def remove_stopwords(text_file):
	def preprocess(s):
		words = str(s).lower().decode('utf-8','ignore')
		words = word_tokenize(words)
		words = [w for w in words if not w in stop_words]
		words = [w for w in words if w.isalpha()]
		words = [w for w in words if len(w) > 1]
		words = [PorterStemmer().stem(w) for w in words] 
		fwords = ''
		for w in words:
			fwords = fwords + PorterStemmer().stem(w) +' '
		return fwords
	
	output = open('../../mq_result/stat/new_matrix/data/text/comment_rem_stem', 'w+')
	file=open(text_file,"r")
	for line in file.readlines():
		texts = preprocess(line)
		print texts
		output.write(texts.encode('utf8') + '\n')
	output.close()
	
def tagger(text_file):
	def preprocess(s):
		words = str(s).lower().decode('utf-8','ignore')
		#words = str(s).decode('utf-8','ignore')
		words = word_tokenize(words)
		words = [w for w in words if w.isalpha()]
		words = [w for w in words if not w in stop_words]
		
		#words = [nltk.pos_tag(w) for w in words if w.isalpha()]
		words = nltk.pos_tag(words)
		words = [(w[0].encode('utf-8'),w[1]) for w in words]

		#words = [w[0].encode('utf-8') for w in words if ('NN' or 'NNS' or 'NNP' or 'NNPS') in w]
		
		return words
		#words = [w for w in words if len(w) > 1]
		#words = [PorterStemmer().stem(w) for w in words] 
		'''
		fwords = ''
		for w in words:
			fwords = fwords + PorterStemmer().stem(w) +' '
		return fwords
		'''
		
	#output = open('../../mq_result/stat/new_matrix/data/text/title_comment_rem_stem', 'w+')
	output = open('./title_comment_lower_tag', 'w+')
	file=open(text_file,"r")
	for line in file.readlines():
		texts = preprocess(line)
		#print texts
		output.write(str(texts) + '\n')
	output.close()	

def count_nn(qstat_file):
	
	qstat_file = './title_comment_lower_tag' 
	data = open(qstat_file)
	qstat = [eval(e.strip()) for e in data]
	#print qstat
	vote = []
	for q in qstat:
		#print q
		#vote.append(q[0]+'_'+q[1])
		for i in q:
			vote.append(i)
			#vote.append(i[0]+'_'+i[1])
		
	
	for i in Counter(vote).most_common():
		a = [i[0],i[1]]
		print(a)	

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
		
def main():
	#text_file = '../../mq_result/stat/corpus'
	#text_file = '../../mq_result/stat/new_matrix/data/text/comment_origin'
	text_file = '../../mq_result/stat/new_matrix/data/text/comment_rem_stem'
	#remove_stopwords(text_file)
	count(text_file)
	'''
	tagger(text_file)
	qstat_file = ''
	count_nn(qstat_file)
	'''
	#querylist = ''
	#username = ''
	
	'''
    userfile =data_path + 'topic_matric_origin.csv'
    df = pd.read_csv(userfile)
    querylist = df.columns.values[1:-1]
    querylist = [q.strip().decode('utf-8','ignore')  for q in list(querylist)]
    username = df.ix[:,'user_topic']
    username = [u.strip().decode('utf-8','ignore')  for u in list(username)]
    context(querylist,username)
	'''

if __name__ == '__main__':
	main()