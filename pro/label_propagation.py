#coding=utf-8
__author__ = 'llt'
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
# return k neighbors index
def navie_knn(dataSet,query,k):
   query = query.reshape((1,-1))
   squaredDist = euclidean_distances(dataSet,query) #计算相似性
   squaredDist = map(float,squaredDist)
   sortedDistIndices=np.argsort(squaredDist) ##返回的是数组值从小到大的索引值
   if k>len(sortedDistIndices):
        k=len(sortedDistIndices)
   return sortedDistIndices[1:k+1]
# build a big graph (normalized weight matrix)

def buildGraph(MatX,kernel_type,rbf_sigma=None,knn_num_neighbors=None):
   num_samples=MatX.shape[0]  ##所有用户数
   affinity_matrix=np.zeros((num_samples,num_samples),np.float32)
   if kernel_type=='rbf':
        if rbf_sigma==None:
            raise ValueError('You should input a sigma of rbf kernel!')
        sim = euclidean_distances(MatX,MatX) #计算相似性
        affinity_matrix=np.exp(sim/(-2.0*rbf_sigma**2))
        for i in xrange(num_samples):
		   row_sum=0.0
		   for j in xrange(num_samples):
			   row_sum+=affinity_matrix[i][j]
		   affinity_matrix[i][:]/=row_sum
   elif kernel_type=='knn':
	   if knn_num_neighbors==None:
		   raise ValueError('You should input a k of knn kernel!')
	   for i in xrange(num_samples):
		   k_neighbors=navie_knn(MatX,MatX[i,:],knn_num_neighbors)  ##某一用户的k近邻索引
		   affinity_matrix[i][k_neighbors]=1.0/knn_num_neighbors
   else:
	   raise NameError('Not support kernel type! You can use knn or rbf!')
   return affinity_matrix
# label propagation
def labelPropagation(Mat_Label,Mat_Unlabel,labels,kernel_type='rbf',rbf_sigma=1.5,\
                   knn_num_neighbors=10,max_iter=10000,tol=1e-2):
   # initialize
   num_label_samples=Mat_Label.shape[0]
   num_unlabel_samples=Mat_Unlabel.shape[0]
   num_samples=num_label_samples+num_unlabel_samples
   labels_list=np.unique(labels)
   num_classes=len(labels_list)
   ##训练数据稀疏标签只有一类
   if num_classes == 1:
       num_classes = 2

   MatX=np.vstack((Mat_Label,Mat_Unlabel))
   clamp_data_label=np.zeros((num_label_samples,num_classes),np.float32)
   for i in xrange(num_label_samples):
       clamp_data_label[i][labels[i]]=1.0
   label_function=np.zeros((num_samples,num_classes),np.float32)
   label_function[0:num_label_samples]=clamp_data_label
   label_function[num_label_samples:num_samples]=-1  ##全部初始化为反例
   # graph construction
   affinity_matrix=buildGraph(MatX,kernel_type,rbf_sigma,knn_num_neighbors)  ##所有点的k近邻
   # print label_function
   #print affinity_matrix
   # start to propagation
   iter=0;pre_label_function=np.zeros((num_samples,num_classes),np.float32)
   changed=np.abs(pre_label_function-label_function).sum()
   while iter<max_iter and changed>tol:
       if iter%1==0:
           print"---> Iteration %d/%d, changed: %f"%(iter,max_iter,changed)
       pre_label_function=label_function
       iter+=1
       # propagation
       label_function=np.dot(affinity_matrix,label_function)
       # clamp
       label_function[0:num_label_samples]=clamp_data_label
       # check converge
       # print label_function
       changed=np.abs(pre_label_function-label_function).sum()
   # get terminate label of unlabeled data
   unlabel_data_labels=np.zeros(num_unlabel_samples)
   for i in xrange(num_unlabel_samples):
       # unlabel_data_labels[i]=np.argmax(label_function[i+num_label_samples])
       ##返回预测为正类的概率归一化
       # print label_function[i+num_label_samples]
       # unlabel_data_labels[i]= (1 + label_function[i+num_label_samples][1] - label_function[i+num_label_samples][0]) * 0.5
       unlabel_data_labels[i]= label_function[i+num_label_samples][1]

       # if unlabel_data_labels[i]<0.5:
       #     print unlabel_data_labels[i]
       # if unlabel_data_labels[i]>=0.5:
       #     continue
       # else:
       #     unlabel_data_labels[i] = -1 *(1-unlabel_data_labels[i])

   return unlabel_data_labels

