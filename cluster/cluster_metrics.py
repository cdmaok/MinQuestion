#coding=utf-8
__author__ = 'llt'
from random import choice
import numpy as np
import pandas as pd
import sklearn.metrics
import csv
import sys
sys.path.insert(0,'..')
import config
result_path = config.result_path
data_path = config.pro_data_path

def read_data(filename):
    data = [eval(e.strip()) for e in open(filename).readlines() ]
    return data

def get_cluster_query(filename):
    dic = {}
    line_num = 0
    with open(filename) as f:
        for line in f:
            line_num += 1
            if line_num > 3:
                line = eval(line.strip())
                if int(line[0]) not in dic:
                    dic[int(line[0])] = [line[1]]
                else:
                    dic[int(line[0])].append(line[1])
    return dic

def get_query_cluster(filename):
    dic = {}
    line_num = 0
    with open(filename) as f:
        for line in f:
            line_num += 1
            if line_num > 3:
                line = eval(line.strip())
                dic[line[1]] = [int(line[0])]
    return dic

def get_gt_query_cluster(gt_yes,gt_no):
    ##return each query belong to which cluster
    dic = {}
    for iter,yes in enumerate(gt_yes):
        for i in range(len(yes)):
            dic[yes[i][0]] = iter

    for iter,no in enumerate(gt_no):
        for i in range(len(no)):
            dic[no[i][0]] = iter
    return dic

def get_gt_cluster_query(gt_yes,gt_no):
    dic = {}
    for iter,yes in enumerate(gt_yes):
        for i in range(len(yes)):
            if iter not in dic:
                dic[iter] = [yes[i][0]]
            else:
                dic[iter].append(yes[i][0])

    for iter,no in enumerate(gt_no):
        for i in range(len(no)):
            dic[iter].append(no[i][0])
    return dic

def get_purity(cluster,gt_cluster,len_gt):
    gt_query = gt_cluster.keys()
    sum = 0.
    num_cluster = 0.
    for key in cluster:
        num_cluster += len(cluster[key])
        num = [0 for i in range(len_gt)]
        for item in cluster[key]:
            if item in gt_query:
                num[gt_cluster[item]] += 1
        sum += max(num)
        # print(num.index(max(num)))
    return sum / len(gt_query)
    # print(sum/num_cluster)

# def get_pr(query_cluster,gt_query_cluster):  #
#     gt_query = gt_query_cluster.keys()
#     label_true = []
#     label_pred = []
#     for i in range(len(gt_query)):
#         label_true.append(gt_query_cluster[gt_query[i]])
#         label_pred.append(query_cluster[gt_query[i]])
#     p = sklearn.metrics.precision_score(label_true,label_pred)
#     r = sklearn.metrics.recall_score(label_true,label_pred)
#     f = sklearn.metrics.f1_score(label_true,label_pred)
#     print(p,r,f)

def get(query_cluster,gt_query_cluster):
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    gt_query = gt_query_cluster.keys()
    for i in range(len(gt_query)):
        for j in range(i+1,len(gt_query)):
            same_class = (gt_query_cluster[gt_query[i]] == gt_query_cluster[gt_query[j]])
            same_cluster = (query_cluster[gt_query[i]] == query_cluster[gt_query[j]])
            if same_class and same_cluster:
                TP += 1
            elif same_class and not same_cluster:
                FN += 1
            elif not same_class and same_cluster:
                FP += 1
            elif not same_class and not same_cluster:
                TN += 1
    return TP,FN,FP,TN

def get_metrics(TP,FN,FP,TN):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F = P*R / (P + R)
    RI = (TP + TN) / (TP + FN + FP + TN)
    return P,R,F,RI

def main():




    gt_yes = read_data(result_path + 'stat/new_matrix/data/gt_yes_0412')
    gt_no = read_data(result_path + 'stat/new_matrix/data/gt_no_0412')
    gt_query_cluster = get_gt_query_cluster(gt_yes,gt_no)

    filenames = ['ap_me2','ap_pm','ap_tcomment','ap_tcomment_m2','ap_tcomment_m3','tcs_km_10','tcs_km_121']
    for filename in filenames:
        path = '/home/yangying/MinQuestion/cluster/'+ filename
        cluster_query = get_cluster_query(path)
        Purity = get_purity(cluster_query,gt_query_cluster,len(gt_yes))

        query_cluster = get_query_cluster(path)
        TP,FN,FP,TN = get(query_cluster,gt_query_cluster)
        P,R,F,RI = get_metrics(TP,FN,FP,TN)
        print(filename)
        print("Purity:",Purity)
        print("Precision:",P)
        print("Recall:",R)
        print("F-measure:",F)
        print("Rand Index:",RI)



if __name__=="__main__":
    main()



