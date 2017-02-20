#coding=utf-8
import os
import merge
import pandas as pd
import sampling_method
import classifier
import sys
import fill
import config
from knntext import text_sim
from fill import mc
data_path = config.data_path
result_path = config.result_path

def probability(rule,probs_file,two_party=False):
	merge.get_probs_file(rule,probs_file,two_party)

def fill_matrix(origin_file,probs_file,goal_fill,origin_fill,fill_method,threshold):
			
	df = pd.read_csv(origin_file)	
	#df = fill.fill(rule,df,fill_method,threshold)	
	#df = mc.fill_whole(fill_method,df)
	df = text_sim.fill_whole(fill_method,df)
	df.to_csv(origin_fill,index=False)		
	merge.get_goal_file(probs_file,goal_fill,origin_fill)


def main():
	
	rule = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}	
	filename = result_path + 'white_old'
	#rule = {'Gender':'Female'}	
	#filename = result_path + 'women'
	
	vote_matrix = 'topic_matric_origin.csv'
	#vote_matrix ='topic_matric_origin_balan.csv'
	#vote_matrix = 'topic_matric_twoparty_balan.csv'
	two_party = False		
	
	fill_method = text_sim.fill_knn_whole
	fill_method_name = 'knntext'
	threshold = 0	
	
	probs_file = filename +  '.pro'
	if not os.path.exists(probs_file):
		probability(rule,probs_file,two_party)
		
	origin_file = data_path + vote_matrix
	vote_matrix = vote_matrix.replace('topic_matric','')		
	origin_fill = filename + '_'+ fill_method_name + str(threshold) + vote_matrix
	goal_file = filename + '_goal' + vote_matrix
	goal_fill = filename +'_'+fill_method_name + str(threshold)+ '_goal'+vote_matrix
	
	if not os.path.exists(goal_fill):
		fill_matrix(origin_file,probs_file,goal_fill,origin_fill,fill_method,threshold)
	
	if not os.path.exists(goal_file):			
		merge.get_goal_file(probs_file,goal_file,origin_file)
	
	####### 2. begin ensemble_FeatureSelection
	f_size = 10
	en = False
	time =1	
	#fs_method_list = [4,5,6,3,0,2,1,8,9,7]
	fs_method_list = [4]
	#[0 svmvoter, 1 lassovoter, 2 dtvoter, 3 Kbesetvoter,
	# 4 sampling_method.EntropyVoterSimple, 5 VarianceVoter, 6 CorelationVoter, 7 WrapperVoter, 8 RndLassovoter, 9 GBDTvoter]
	
	origin_file = origin_fill    #choose fill>
	goal_file = goal_fill 
	
	print '------',time,' time------fill:'+fill_method_name+' '+str(threshold)+'-pos ---ensemble:',en,'------'
	print goal_file
	for method_type in fs_method_list:
		if(en):
			feature = sampling_method.emsemble_sampling(time,en,probs_file,origin_file,method_type,f_size)
		else:
			feature = sampling_method.emsemble_sampling(time,en,probs_file,goal_file,method_type,f_size)
		classifier.main(feature,goal_file,f_size)
	
		

if __name__ == '__main__':
	main()
