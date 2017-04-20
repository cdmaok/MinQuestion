#coding=utf-8
import numpy as np
import pandas as pd
import sys
import ast
import json
from nltk.stem.porter import PorterStemmer
import os
sys.path.append("..")
import config
import merge
from sklearn.metrics.pairwise import cosine_similarity
from knntext.doc2vec import getvector
from sklearn.feature_extraction.text import CountVectorizer
import RAKE
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from collections import OrderedDict
stop_words = stopwords.words('english')

data_path = config.data_path
result_path = config.result_path
Comment_Dir = config.Comment_Dir
Filemodel = config.Filemodel
Stoplist = config.Stoplist

def context(querylist,username):
    directory = Comment_Dir
    data = [[] for i in xrange(len(querylist))]
    for filename in os.listdir(directory):
        filepath = directory + '/' + filename
        lines = [l.strip().decode('utf-8','ignore') for l in open(filepath).readlines()]
        for l in lines:
            l = l.replace('\\','/')
            dic = json.loads(l,strict= False)
            title = dic[u'title'].strip().encode('utf-8','ignore')
            text = dic[u'text'].strip().encode('utf-8','ignore')
            user = dic[u'name'].strip().encode('utf-8','ignore')
            point = dic[u'point'].strip().encode('utf-8','ignore')
            if title in querylist and user in username:
                ind = querylist.index(title)
                if data[ind] == []:
                    data[ind] = [[ind,title]]
                data[ind].append([user,point,text])
    for i in xrange(len(querylist)):
        if data[i] == []:
            data[i] = [[i,querylist[i].encode('utf-8','ignore')]]
        print data[i]

def get_user(pro_file):
    username = []
    probs = [eval(e.strip()) for e in open(pro_file).readlines() ]
    for e in probs:
        if e[1] == 1:
            username.append(e[0])
    return username

def read_data(filename):
    datas = [eval(e.strip()) for e in open(filename).readlines() ]
    return datas

def get_query_comment(data,user):
    query_yes = OrderedDict()
    query_no = OrderedDict()
    for i in data:
        query = i[0][1]
        query_yes[query] = query
        query_no[query] = query
        if len(i) > 1:
            for j in xrange(1,len(i)):
                if i[j][0] in user:
                    if i[j][1] == 'yes':
                        query_yes[query] += ' ' + i[j][2]
                    elif i[j][1] == 'no':
                        query_no[query] += ' ' + i[j][2]
    return query_yes,query_no

def get_corpus(query_yes,query_no):
    qt_set = []
    for key in query_yes:
        qt_set.append(query_yes[key])
    for key in query_no:
        qt_set.append(query_no[key])
    return qt_set

def tf_idf(query_yes,query_no):
    def preprocess(s): 
        words = str(s).lower().decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if not w in stop_words]
        words = [w for w in words if w.isalpha()]
        words = [w for w in words if len(w)>1]
        words = [PorterStemmer().stem(w) for w in words] 
        return words

    documents = get_corpus(query_yes,query_no)
    texts = [[word for word in preprocess(document)] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf

def compute_tfidf(query_yes,query_no,filename):
    corpus_tfidf = tf_idf(query_yes,query_no)
    dic = {}
    for i in range(len(query_yes)):
        for j in range(i+1,len(query_no)):
            if(corpus_tfidf[i]==[] or corpus_tfidf[j]==[]):
              dic[(i,j)] = (i,j,0,0,0,0)
            else:
              c = [corpus_tfidf[i],corpus_tfidf[j]]
              index = similarities.MatrixSimilarity(c)
              a = [ n.tolist() for n in index]
              tf_yy = a[0][1]
              c = [corpus_tfidf[i+len(query_no)],corpus_tfidf[j+len(query_no)]]
              index = similarities.MatrixSimilarity(c)
              a = [ n.tolist() for n in index]
              tf_nn = a[0][1]
              c = [corpus_tfidf[i],corpus_tfidf[j+len(query_no)]]
              index = similarities.MatrixSimilarity(c)
              a = [ n.tolist() for n in index]
              tf_yn = a[0][1]
              c = [corpus_tfidf[i+len(query_no)],corpus_tfidf[j]]
              index = similarities.MatrixSimilarity(c)
              a = [ n.tolist() for n in index]
              tf_ny = a[0][1]
              dic[(i,j)] = (i,j,tf_yy,tf_nn,tf_yn,tf_ny)
    output = sys.stdout
    outputfile = open(result_path + 'stat/new_matrix/' + filename,'w')
    sys.stdout = outputfile
    print("(i,j,sim_yy,sim_nn,sim_yn,sim_ny)")
    print(dic)
    outputfile.close()
    sys.stdout = output

	
def get_part_comment(group,f_query_pair,filename):
  
    #group = 'stat/alluser.pro'
    #userfile =data_path+ 'topic_matric_origin.csv'
    #df = pd.read_csv(userfile)
    #user = df.ix[:,'user_topic']
    #user = [u.strip().decode('utf-8','ignore')  for u in list(user)]
    user = read_data('/home/yangying/MinQuestion/cluster/user')
    user = user[0]
    #print user
    query_user_comment = result_path + 'stat/new_matrix/data/query_user_vote_comment'
    datas = read_data(query_user_comment)
    query_yes,query_no = get_query_comment(datas,user)
  
   
    ##compute similarity
    method = 'tfidf_'
    #filename = method + group.split('/')[-1].replace('.pro','')
    compute_tfidf_part(query_yes,query_no,filename,f_query_pair)

	
def compute_tfidf_part(query_yes,query_no,filename,f_query_pair):
    corpus_tfidf = tf_idf(query_yes,query_no)
    dic = {}
	
    #f_query_pair = './cluster/gt.txt'
    query_pair = read_data(f_query_pair)
	
    for q in query_pair:
		i = q[1]
		j = q[2]
		if(corpus_tfidf[i]==[] or corpus_tfidf[j]==[]):
		  dic[(i,j)] = (i,j,0,0,0,0)
		else:
		  c = [corpus_tfidf[i],corpus_tfidf[j]]
		  index = similarities.MatrixSimilarity(c)
		  a = [ n.tolist() for n in index]
		  tf_yy = a[0][1]
		  c = [corpus_tfidf[i+len(query_no)],corpus_tfidf[j+len(query_no)]]
		  index = similarities.MatrixSimilarity(c)
		  a = [ n.tolist() for n in index]
		  tf_nn = a[0][1]
		  c = [corpus_tfidf[i],corpus_tfidf[j+len(query_no)]]
		  index = similarities.MatrixSimilarity(c)
		  a = [ n.tolist() for n in index]
		  tf_yn = a[0][1]
		  c = [corpus_tfidf[i+len(query_no)],corpus_tfidf[j]]
		  index = similarities.MatrixSimilarity(c)
		  a = [ n.tolist() for n in index]
		  tf_ny = a[0][1]
		  dic[(i,j)] = (i,j,tf_yy,tf_nn,tf_yn,tf_ny)
    output = sys.stdout
    #outputfile = open('/home/yangying/MinQuestion/cluster/' + filename,'w')
    outputfile = open(filename,'w')
    sys.stdout = outputfile
    print("(i,j,sim_yy,sim_nn,sim_yn,sim_ny)")
    print(dic)
    outputfile.close()
    sys.stdout = output	
	
def relevant_doc2vec(query_yes,query_no):
    vectors_yes = [list(getvector(query_yes[key].decode('utf-8','ignore'),Filemodel)) for key in query_yes]
    vectors_no = [list(getvector(query_no[key].decode('utf-8','ignore'),Filemodel)) for key in query_no]
    yes = np.asarray(vectors_yes)
    no = np.asarray(vectors_no)
    sim_yy = cosine_similarity(yes,yes)
    sim_nn = cosine_similarity(no,no)
    sim_yn = cosine_similarity(yes,no)
    sim_ny = cosine_similarity(no,yes)
    return sim_yy,sim_nn,sim_yn,sim_ny

def relevant_rake(query_yes,query_no):
    vectors_yes = [query_yes[key].decode('utf-8','ignore') for key in query_yes]
    vectors_no = [query_no[key].decode('utf-8','ignore') for key in query_no]
    Rake = RAKE.Rake(Stoplist)
    corpus_yes = []
    corpus_no = []
    for q in vectors_yes:
        keyword = Rake.run(q) ##return keyword
        corpus_yes.append(' '.join([t[0].replace('-',' ') for t in keyword])) #merge keyword
    #countvect = CountVectorizer()
    for q in vectors_no:
        keyword = Rake.run(q) ##return keyword
        corpus_no.append(' '.join([t[0].replace('-',' ') for t in keyword])) #merge keyword
    countvect = CountVectorizer()
    
    yes = countvect.fit_transform(corpus_yes).toarray()
    no = countvect.fit_transform(corpus_no).toarray()
    #print(len(yes[0]),len(no[0]))

    sim_yy = cosine_similarity(yes,yes)
    sim_nn = cosine_similarity(no,no)

    sim_yn = [[0 for j in xrange(len(vectors_no))] for i in xrange(len(vectors_yes))]
    for i in xrange(len(vectors_yes)):
        corpus = []
        for j in xrange(len(vectors_no)):
            if j == i:
              corpus.append(corpus_yes[j])
            else:
              corpus.append(corpus_no[j])
        yn = countvect.fit_transform(corpus).toarray()
        for j in xrange(len(yn)):
          sim_yn[i][j] = cosine_similarity(yn[i].reshape(1,-1),yn[j].reshape(1,-1))[0][0]

    sim_ny = [[0 for j in xrange(len(vectors_yes))] for i in xrange(len(vectors_no))]
    for i in xrange(len(vectors_no)):
        corpus = []
        for j in xrange(len(vectors_yes)):
            if j == i:
              corpus.append(corpus_no[j])
            else:
              corpus.append(corpus_yes[j])
        ny = countvect.fit_transform(corpus).toarray()
        for j in xrange(len(ny)):
          sim_ny[i][j] = cosine_similarity(ny[i].reshape(1,-1),ny[j].reshape(1,-1))[0][0]
    return sim_yy,sim_nn,np.array(sim_yn),np.array(sim_ny)

def print_sim(sim_yy,sim_nn,sim_yn,sim_ny,filename):
    output = sys.stdout
    outputfile = open(result_path + 'stat/new_matrix/' + filename,'w')
    sys.stdout = outputfile
    print("(i,j,sim_yy,sim_nn,sim_yn,sim_ny)")

    for i in xrange(sim_yy.shape[0]):
        for j in xrange(i+1,sim_yy.shape[0]):
            print (i,j,sim_yy[i][j],sim_nn[i][j],sim_yn[i][j],sim_ny[i][j])
    outputfile.close()
    sys.stdout = output


def main():
    ##get query_user_comment
    # userfile =data_path+ 'topic_matric_origin.csv'
    # df = pd.read_csv(userfile)
    # querylist = df.columns.values[1:-1]
    # querylist = [q.strip().decode('utf-8','ignore')  for q in list(querylist)]
    #username = df.ix[:,'user_topic']
    #username = [u.strip().decode('utf-8','ignore')  for u in list(username)]
    #context(querylist,username)

    ##stu
    #group = 'stat/stu.pro'  ##stu
    #group = 'stat/white.pro'  ##white
    #goal_pro = result_path + group
    #user = get_user(goal_pro)

    ##alluser
    group = 'stat/alluser.pro'
    userfile =data_path+ 'topic_matric_origin.csv'
    df = pd.read_csv(userfile)
    user = df.ix[:,'user_topic']
    user = [u.strip().decode('utf-8','ignore')  for u in list(user)]

    #query_user_comment = result_path + 'stat/new_matrix/query_user_comment_twoparty_balan'
    query_user_comment = result_path + 'stat/new_matrix/data/query_user_vote_comment'
    datas = read_data(query_user_comment)
    query_yes,query_no = get_query_comment(datas,user)
  
   
    ##compute similarity
    method = 'tfidf_'
    filename = method + group.split('/')[-1].replace('.pro','')
    #compute_tfidf(query_yes,query_no,filename)
    compute_tfidf_part(query_yes,query_no,filename)
    #sim_yy,sim_nn,sim_yn,sim_ny = relevant_doc2vec(query_yes,query_no)
    #method = 'doc2vec_'
    #filename = method + group.split('/')[-1].replace('.pro','')
    #print_sim(sim_yy,sim_nn,sim_yn,sim_ny,filename)

    # sim_yy,sim_nn,sim_yn,sim_ny = relevant_rake(query_yes,query_no)
    # method = 'rake_'
    # filename = method + group.split('/')[-1].replace('.pro','')
    # print_sim(sim_yy,sim_nn,sim_yn,sim_ny,filename)


if __name__ == '__main__':
    main()

