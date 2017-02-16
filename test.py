#coding=utf-8
import merge
import pandas as pd
import sampling_method
import classifier
import sys
sys.path.append('./fill')
import fill
import mc

data_path = '../mq_data/'
result_path = '../mq_result/'

def main():
	
	######## 1 prepare 4 files (1 probs + 1 fill + 2 goal), just change the rule and filename 
	### 1.1 get probs_file 
	#change line20 vote_matrix,line21 two_party,line 23 rule,line 24 filename, line25 fill_method,line26 threshold,line43 fill_xxx_whole
	
	#vote_matrix = 'topic_matric_origin.csv'
	#vote_matrix ='topic_matric_origin_balan.csv'
	vote_matrix = 'topic_matric_twoparty_balan.csv'
	two_party = True
	
	rule = {'Ethnicity':'White','Age':[40,50,60,70,80,90,100,110]}	
	filename = result_path + 'white_old'			
	fill_method= 'knn'
	threshold = 0	
			
	probs_file = filename +  '.pro'
	merge.get_probs_file(rule,probs_file,two_party)
	
	### 1.2 get goal_file and goal_fill_file		
	origin_file = data_path + vote_matrix
	vote_matrix = vote_matrix.replace('topic_matric','')		
	origin_fill = filename + '_'+ fill_method + str(threshold) + vote_matrix
	goal_file = filename + '_goal' + vote_matrix
	goal_fill = filename +'_'+fill_method + str(threshold)+ '_goal'+vote_matrix
	
	df = pd.read_csv(origin_file)	
	#df = fill.fill(rule,df,fill_method,threshold)	
	df = mc.fill_whole(mc.fill_knn_whole,df)
	df.to_csv(origin_fill,index=False)		
	merge.get_goal_file(probs_file,goal_fill,origin_fill)	
	
	####### 2. begin ensemble_FeatureSelection
	#change line48 feature_size,line49 time,line50 ensemble,line51 fs_method_list	
	f_size = 10
	en = True
	time =10		
	fs_method_list = [4,5,6,3,0,2,1,8,9,7]
	#type_list = [0,1,2,3,4,5,6,7,8,9]
	#type_list = [7]
	#[0 svmvoter, 1 lassovoter, 2 dtvoter, 3 Kbesetvoter,
	# 4 sampling_method.EntropyVoterSimple, 5 VarianceVoter, 6 CorelationVoter, 7 WrapperVoter, 8 RndLassovoter, 9 GBDTvoter]
	
	origin_file = origin_fill    #choose fill>
	goal_file = goal_fill 
	
	print '------',time,' time------fill:'+fill_method+' '+str(threshold)+'----ensemble:',en,'------'
	print goal_file
	for method_type in fs_method_list:
		if(en):
			feature = sampling_method.emsemble_sampling(time,en,probs_file,origin_file,method_type,f_size)
		else:
			feature = sampling_method.emsemble_sampling(time,en,probs_file,goal_file,method_type,f_size)
		classifier.main(feature,goal_file,f_size)
	
		

if __name__ == '__main__':
	main()
