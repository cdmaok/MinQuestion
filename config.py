#coding=utf-8

### this file contains only path
#test.py
Home = '/home/yangying'
data_path = Home+'/mq_data/'
result_path = Home+'/mq_result/'
two_party_flag = False
#vote_matrix = 'topic_matric_twoparty_balan.csv'
#vote_matrix = 'topic_matric_origin.csv'
#vote_matrix = 'topic_matric_twoparty_balan_reduce.csv'
vote_matrix ='topic_matric_origin_balan.csv'

origin_file = data_path + vote_matrix


#pro/pro.py
pro_path = data_path + 'process_data/'
pro_data_path = pro_path + 'data.csv'
pro_middle_0_path = pro_path + 'middledata_0.csv'
pro_middle_freq_path = pro_path + 'middledata_frequent.csv'

#pro/pro_twoparty.py
protp_data_path = pro_path + 'data_twoparty.csv'
protp_middle_0_path = pro_path + 'middledata_0_twoparty.csv'
protp_middle_freq_path = pro_path + 'middledata_frequent_twoparty.csv'

#fill/fill.py
Path = data_path
Vote_Matrix = origin_file
Cluster_Result = Path + 'process_data/fulldata_xmeans.csv'
Cluster_Result_tp = Path + 'process_data/fulldata_xmeans_twoparty.csv'
Comment_Dir = Path + 'comments/'
Filemodel = Path + 'wiki_doc2vec.bin'
Stoplist = Path + 'SmartStoplist.txt'


#knntext/text_sim.py
#Text_path = data_path + 'text.csv'
#Text_path_tp = data_path + 'text_twoparty.csv'

#origin_file = data_path+'topic_matric_twoparty_balan.csv'
#origin_file = data_path+'topic_matric_origin.csv'


#lp.py
i2int_path = '/home/yangying/mq_data/process_data/data.csv.2int'
fd_path = '/home/yangying/mq_data/process_data/data.csv.fd'

groundtruth = data_path + 'groundtruth/white_old.gt'
