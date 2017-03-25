#coding=utf-8

## Notice: PCA can only extract feature num which is the minimun of sample_num and feature num

import pandas as pd
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS
from skfeature.function.sparse_learning_based import NDFS
from skfeature.function.similarity_based import SPEC
from skfeature.function.sparse_learning_based import UDFS
from skfeature.function.statistical_based import low_variance
from skfeature.utility.sparse_learning import feature_ranking
from skfeature.utility import construct_W
import threading
import numpy as np
import PFA

## get the first n elements of array,but if array[n-1] == array[n],then the (n+1)th element will be return.
def cut(score,index,size):
    src = score[index[size - 1]]
    total = len(score)
    ans = size - 1
    for i in range(size,total):
        tar = score[index[i]]
        if not isEqual(src,tar):
            break
        ans = i
    if size != ans: print 'expanding to' ,ans
    return ans

def isEqual(a1,a2):
    if abs(a1-a2)< 0.001:
        return True
    return False

def getHighScoreIndex(score,size):
    index  = sorted(range(len(score)),key=lambda k:score[k],reverse=True)
    expand = cut(score,index,size)
    return index[0:expand]

def getX(df):
    headers = list(df.columns)
    end = headers.index('Class')
    x = df.ix[:,0:end].as_matrix()
    return x


class lapscore(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.lapscore()

    def lapscore(self):
        X = getX(self.sampled_df)
        #construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs_W)
        # obtain the scores of features
        score = lap_score.lap_score(X, W=W)
        # sort the feature scores in an ascending order according to the feature scores
        idx = lap_score.feature_ranking(score)
        feat = []
        for i in xrange(self.num):
            feat.append(idx[i])
        self.topics = feat ##

    def getTopic(self):
        return self.topics

class mcfs(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.mcfs()

    def mcfs(self):
        X = getX(self.sampled_df)
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)
        # obtain the feature weight matrix
        Weight = MCFS.mcfs(X, n_selected_features=self.num, W=W, n_clusters=20)
        # sort the feature scores in an ascending order according to the feature scores
        idx = MCFS.feature_ranking(Weight)
        feat = []
        for i in xrange(self.num):
            feat.append(idx[i])
        self.topics = feat ##

    def getTopic(self):
        return self.topics

class ndfs(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.ndfs()

    def ndfs(self):
        X = getX(self.sampled_df)
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)
        # obtain the feature weight matrix
        Weight = NDFS.ndfs(X, W=W, n_clusters=20)
        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)
        feat = []
        for i in xrange(self.num):
            feat.append(idx[i])
        self.topics = feat ##

    def getTopic(self):
        return self.topics

class spec(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.spec()

    def spec(self):
        X = getX(self.sampled_df)
         # specify the second ranking function which uses all except the 1st eigenvalue
        kwargs = {'style': 0}
        # obtain the scores of features
        score = SPEC.spec(X, **kwargs)
        # sort the feature scores in an descending order according to the feature scores
        idx = SPEC.feature_ranking(score, **kwargs)
        feat = []
        for i in xrange(self.num):
            feat.append(idx[i])
        self.topics = feat ##

    def getTopic(self):
        return self.topics

class udfs(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.udfs()

    def udfs(self):
        X = getX(self.sampled_df)
         # obtain the feature weight matrix
        Weight = UDFS.udfs(X, gamma=0.1, n_clusters=20)
        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)
        feat = []
        for i in xrange(self.num):
            feat.append(idx[i])
        self.topics = feat ##

    def getTopic(self):
        return self.topics

class lowvariance(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.lowvariance()

    def lowvariance(self):
        X = getX(self.sampled_df)
        # p = 0.1    # specify the threshold p to be 0.1
        # perform feature selection and obtain the dataset on the selected features
        # selected_features = low_variance.low_variance_feature_selection(X, p*(1-p))

        v = list(np.var(X,axis=0))
        score = sorted(range(len(v)),key = lambda i:v[i],reverse=True)
        score = sorted(range(len(v)),key = lambda i:v[i],reverse=True)
        self.topics = getHighScoreIndex(score,self.num)

    def getTopic(self):
        return self.topics

class pfa(threading.Thread):
    def __init__(self,X,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = X
        self.topics = []
        self.num = f_num

    def run(self):
        self.pfa()

    def pfa(self):
        X = getX(self.sampled_df)
        pfa = PFA.PFA(self.num)
        pfa.fit(X)
        # To get the column indices of the kept features
        column_indices = pfa.indices_
        self.topics = column_indices

    def getTopic(self):
        return self.topics

if __name__ == '__main__':
    filename = ('./white_old_sim0_goal_origin_balan.csv')
    df = pd.read_csv(filename)
    # pv = lapscore(df,10)
    # pv.lapscore()

    # pv = mcfs(df,10)
    # pv.mcfs()

    # pv = ndfs(df,10)
    # pv.ndfs()

    # pv = spec(df,10)
    # pv.spec()

    # pv = udfs(df,10)
    # pv.udfs()

    # pv = lowvariance(df,10)
    # pv.lowvariance()

    pv = pfa(df,10)
    pv.pfa()
    print pv.getTopic()