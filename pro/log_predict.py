#coding=utf-8
__author__ = 'llt'
from random import choice
import numpy as np
import pandas as pd
import label_propagation
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import math
data_path = '/home/yangying/mq_data/process_data/data.csv'
middle_0_path = '/home/yangying/mq_data/process_data/middledata_0.csv'
middle_freq_path = '/home/yangying/mq_data/process_data/middledata_frequent.csv'
def attr(): ##属性取值
    df = pd.read_csv(data_path)
    mat = df.as_matrix()
    row,col = mat.shape
    attr = [[] for i in xrange(col)]
    for i in xrange(1,col):
        for j in xrange(row):
            if mat[j][i]==mat[j][i]:
                if mat[j][i] not in attr[i]:
                    attr[i].append(mat[j][i])
        attr[i] = [x for x in attr[i] if attr[i].count(x) == 1]
    return attr,row,col

def getPro(rule):
    lattr,row,col = attr()
    attributes = ['User_name','Age','Location','Gender','President','Ideology','Education','Party','Ethnicity','Relationship','Income','Interested','Occupation','Looking','Religion']
    ind = [attributes.index(key) for key in rule]
    feature = [rule[key] if type(rule[key]) == list else [rule[key]]for key in rule]
    attr_num = len(rule)
    fea = [[] for i in xrange(attr_num)]
    for k in xrange(attr_num):
        for j in feature[k]:
            if j in lattr[ind[k]]:
                fea[k].append(lattr[ind[k]].index(j))
    line_num = 0
    Mat_Label_1 = []
    Mat_Label_0 = []
    Mat_Unlabel = []
    num_1 = 0
    num_0 = 0
    df_1 = pd.read_csv(middle_0_path)
    with open(middle_freq_path) as f:
        for line in f:
            line_num += 1
            if line_num != 1:
                line = line.strip('\n')
                line = line.split(',')
                for j in xrange(1,len(line)):
                    line[j] = float(line[j])
                miss = 0
                for j in xrange(attr_num):
                    # if math.isnan(df_1.iloc[line_num-2,ind[j]]):
                    if df_1.iloc[line_num-2,ind[j]] == 999: ##missing value
                        miss = 1
                if miss == 1: ##missing value
                    # continue
                    tmp = []
                    for j in xrange(attr_num):
                        if line[ind[j]] in fea[j]: ##missing value
                            miss += 1
                        if j ==0:
                            tmp.extend(line[1:ind[j]])
                        else:
                            tmp.extend(line[ind[j-1]+1:ind[j]])
                    tmp.extend(line[ind[j]+1:col])
                    Mat_Unlabel.append(tmp)
                else:
                    miss = 0
                    tmp = []
                    for j in xrange(attr_num):
                        if line[ind[j]] in fea[j]: ##missing value
                            miss += 1
                        if j ==0:
                            tmp.extend(line[1:ind[j]])
                        else:
                            tmp.extend(line[ind[j-1]+1:ind[j]])
                    tmp.extend(line[ind[j]+1:col])
                    if miss == attr_num:
                        Mat_Label_1.append(tmp)
                        num_1 += 1
                    else:
                        Mat_Label_0.append(tmp)
                        num_0 += 1
    # print num_1,num_0

    #predict


    #  # # 测试
    # for i in xrange(10):
    #     Mat_sample = Mat_Label_0[:]
    #     labels_sample = [0 for j in xrange(num_0)]
    #     Mat_sample.extend(Mat_Label_1[:])
    #     tmp = [1 for j in xrange(num_1)]
    #     labels_sample.extend(tmp)
    #     if num_1>num_0:
    #         num = num_0
    #         while num < num_1:
    #             x = choice(Mat_Label_0)
    #             Mat_sample.append(x)
    #             labels_sample.append(0)
    #             num += 1
    #     else:
    #         num = num_1
    #         while num < num_0:
    #             x = choice(Mat_Label_1)
    #             Mat_sample.append(x)
    #             labels_sample.append(1)
    #             num += 1
    #     Mat_Unlabel = []
    #     # 测试集
    #     test_labels = [] #test' labels
    #     SUM = len(Mat_sample)
    #     for j in xrange(int(SUM * 0.2)):
    #         x = choice(Mat_sample)
    #         test_labels.append(labels_sample[Mat_sample.index(x)])
    #         Mat_Unlabel.append(x)
    #         del labels_sample[Mat_sample.index(x)]
    #         del Mat_sample[Mat_sample.index(x)]

    # Mat_Label = np.array(Mat_sample)
    # Mat_Unlabel = np.array(Mat_Unlabel)
    # labels = np.array(labels_sample)
    # test_labels = np.array(test_labels)

    Mat_sample = Mat_Label_0[:]
    labels_sample = [0 for j in xrange(num_0)]
    Mat_sample.extend(Mat_Label_1[:])
    tmp = [1 for j in xrange(num_1)]
    labels_sample.extend(tmp)
    if num_1>num_0:
        num = num_0
        while num < num_1:
            x = choice(Mat_Label_0)
            Mat_sample.append(x)
            labels_sample.append(0)
            num += 1
    else:
        num = num_1
        while num < num_0:
            x = choice(Mat_Label_1)
            Mat_sample.append(x)
            labels_sample.append(1)
            num += 1
    Mat_Label = np.array(Mat_sample)
    Mat_Unlabel = np.array(Mat_Unlabel)
    labels = np.array(labels_sample)


    # num_label_samples=Mat_Label.shape[0]#
    # MatX=np.vstack((Mat_Label,Mat_Unlabel))
    #
    # enc = OneHotEncoder()
    # enc.fit(MatX)
    # Mat_Label = enc.transform(MatX[0:num_label_samples]).toarray() ##encode
    # Mat_Unlabel = enc.transform(MatX[num_label_samples:]).toarray() ##encode
    # print Mat_Label[0]
    # print Mat_Unlabel[0]

    train_cols = Mat_Label
    logit = sm.Logit(labels, train_cols)
    result = logit.fit()
    tmp = result.predict(Mat_Unlabel)
    probs = []
    num = 0
    line_num = 0
    with open(middle_0_path) as f:
        for line in f:
            line_num += 1
            if line_num != 1:
                line = line.strip('\n')
                line = line.split(',')
                for j in xrange(1,len(line)):
                    line[j] = float(line[j])
                miss = 0
                for j in xrange(attr_num):
                  # if math.isnan(line[ind[j]]):
                    if line[ind[j]] == 999: ##missing value
                        miss = 1
                if miss == 1: ##missing value
                    probs.append(round((tmp[num]),2))
                    num += 1
                else:
                    miss = 0
                    for j in xrange(attr_num):
                        if line[ind[j]] in fea[j]: ##missing value
                            miss += 1
                    if miss == attr_num:
			            probs.append(1)
                    else:
			            probs.append(0)
    # print tmp
    df = pd.read_csv(data_path)
    names = list(df['User_name'])
    print len(names) == len(probs)
    tt = len(names)
    temp = [ (names[i],probs[i]) for i in range(tt)]
    return temp
        # train_cols = Mat_Label
        # logit = sm.Logit(labels, train_cols)
        # result = logit.fit()
        # tmp = result.predict(Mat_Unlabel)
        # matchCount = 0
        # print tmp
        # for j in xrange(len(tmp)):
        #     predict = tmp[j] > 0.5
        #     if predict == bool(test_labels[j]):
        #         matchCount += 1
        # accuracy = float(matchCount) / len(tmp)
        # print accuracy




# main function
if __name__=="__main__":


    rule = {'Education':['High School','Some College','Associates Degree'],'Income':['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000']}
    # rule = {'Education':['High School','Some College','Associates Degree']}

    print getPro(rule)

