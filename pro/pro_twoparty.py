#coding=utf-8
__author__ = 'llt'
from random import choice
import numpy as np
import pandas as pd
import label_propagation
from sklearn.preprocessing import OneHotEncoder
data_path = '/home/yangying/mq_data/process_data/data_twoparty.csv'
middle_0_path = '/home/yangying/mq_data/process_data/middledata_0_twoparty.csv'
middle_freq_path = '/home/yangying/mq_data/process_data/middledata_frequent_twoparty.csv'
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
    #print num_1,num_0

    #predict
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


    # Mat_Label = np.array(Mat_sample)
    # Mat_Unlabel = np.array(Mat_Unlabel)
    # labels = np.array(labels_sample)

    Mat_Label = np.array(Mat_sample)
    Mat_Unlabel = np.array(Mat_Unlabel)
    labels = np.array(labels_sample)

    num_label_samples=Mat_Label.shape[0]

    MatX=np.vstack((Mat_Label,Mat_Unlabel))
    enc = OneHotEncoder()
    enc.fit(MatX)
    Mat_Label = enc.transform(MatX[0:num_label_samples]).toarray() ##encode
    Mat_Unlabel = enc.transform(MatX[num_label_samples:]).toarray() ##encode

    unlabel_data_labels = label_propagation.labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.5,max_iter=1000)

    line_num = 0
    num = 0
    probs = []
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
                    if line[ind[j]] == 999: ##missing value
                        miss = 1
                if miss == 1: ##missing value
                    probs.append(round((unlabel_data_labels[num]),2))
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
    df = pd.read_csv(data_path)
    names = list(df['User_name'])
    print len(names) == len(probs)
    tt = len(names)
    tmp = [ (names[i],probs[i]) for i in range(tt)]
    #print tmp
    return tmp
	
# main function
if __name__=="__main__":
    #df = pd.read_csv('./data.csv')
    #attr = attr()


    # Hillary = ['California','Washington DC','Hawaii','Maryland','Massaehusetts','New York','Rhode Island','Vermont','Connecticut',
    #            'Delaware','Illinois','Maine','New Jersey','New Mexico','Oregon','Washington','Colorado','Minnesota','Nevada','New Hampshire','Virginia']
    # Tramp = ['Alabama','Alaska','Arkansas','Idaho','Kansas','Kentucky','Louisiana','Mississippi','Montana','Nebraska','North Dakota','Oklahoma','South Carolina','South Dakota','Tennessee','Texas','Utah','West Virginia','Wyoming',
    #          'Michigan','Florida','Lowisiana','Ohio','Pennsyirania','Wisconsin','Arizona','Georgia','Indiana','Missour','North Carolina']


    '''
	attr_num = 2 ##属性个数
    ind = [0 for j in xrange(attr_num)]
    ind[0] = attributes.index('Education')
    ind[1] = attributes.index('Income')
    feature = [0 for j in xrange(attr_num)]
    # feature = [['California','Washington DC','Hawaii','Maryland','Massaehusetts','New York','Rhode Island','Vermont','Connecticut','Delaware','Illinois','Maine','New Jersey','New Mexico','Oregon','Washington','Colorado','Minnesota','Nevada','New Hampshire','Virginia']]  #Hillary必胜洲
    # feature = ['Alabama','Alaska','Arkansas','Idaho','Kansas','Kentucky','Louisiana','Mississippi','Montana','Nebraska','North Dakota','Oklahoma','South Carolina','South Dakota','Tennessee','Texas','Utah','West Virginia','Wyoming'] #Tramp必胜洲
    feature[0] = ['High School','Some College','Associates Degree'] ##low-education
    feature[1] = ['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000'] ##middle-class
	'''

    rule = {'Education':['High School','Some College','Associates Degree'],'Income':['$50,000 to $75,000','$35,000 to $50,000','$25,000 to $35,000','$75,000 to $100,000']}

    print getPro(rule)


    # # 测试
    # accuracy = 0
    # for i in xrange(10):
    #     Mat_Unlabel = []
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
    #     # 测试集
    #     test_labels = [] #test' labels
    #     SUM = len(Mat_sample)
    #     for j in xrange(int(SUM * 0.2)):
    #         x = choice(Mat_sample)
    #         test_labels.append(labels_sample[Mat_sample.index(x)])
    #         Mat_Unlabel.append(x)
    #         del labels_sample[Mat_sample.index(x)]
    #         del Mat_sample[Mat_sample.index(x)]
    #
    #     Mat_Label = np.array(Mat_sample)
    #     Mat_Unlabel = np.array(Mat_Unlabel)
    #     labels = np.array(labels_sample)
    #     test_labels = np.array(test_labels)
    #
    #     unlabel_data_labels = label_propagation.labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.5)
    #     # unlabel_data_labels=labelPropagation(Mat_Label,Mat_Unlabel,labels,kernel_type='knn',knn_num_neighbors=10,max_iter=400)
    #
    #     # 预测正确率
    #     num = 0.
    #     for j in xrange(len(test_labels)):
    #         if unlabel_data_labels[j] == test_labels[j]:
    #                     num += 1
    #     print num / len(test_labels)
    #     accuracy += num / len(test_labels)
    #     print i
    # print accuracy / 10
