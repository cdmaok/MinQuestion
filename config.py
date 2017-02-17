#coding=utf-8

### this file contains only path
#test.py
data_path = '../mq_data/'
result_path = '../mq_result/'

#pro/pro.py
pro_path = '../../mq_data/process_data/'
pro_data_path = pro_path + 'data.csv'
pro_middle_0_path = pro_path + 'middledata_0.csv'
pro_middle_freq_path = pro_path + 'middledata_frequent.csv'

#pro/pro_twoparty.py

protp_data_path = pro_path + 'data_twoparty.csv'
protp_middle_0_path = pro_path + 'middledata_0_twoparty.csv'
protp_middle_freq_path = pro_path + 'middledata_frequent_twoparty.csv'

#fill/fill.py
Path = '../../mq_data'
Cluster_Result = Path + '/process_data/fulldata_xmeans.csv'
Vote_Matrix = Path + '/topic_matric_origin.csv'
Comment_Dir = Path + '/comments/'
Filemodel = Path + '/wiki_doc2vec.bin'
Stoplist = Path + 'SmartStoplist.txt'

#lp.py
i2int_path = '../mq_data/process_data/data.csv.2int'
fd_path = '../mq_data/process_data/data.csv.fd'
