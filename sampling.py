# -*- coding: utf-8 -*-
import pandas as pd
import os
import merge
import config
data_path = config.data_path
result_path = config.result_path

def sampling(probs_file,origin_file):
	#sample_df = df.resample(frac=0.1)
	type = 1
	sample_df = merge.get_sample(probs_file,origin_file,type)
	return sample_df
	
def filter_twoparty_file(probs_file,df,origin_fill,goal_fill):
	df =  pd.read_csv(df,dtype={"user_topic":str,"Class":str})
	#df = df.ix[:,[0,1,2,3,4,5,-1]]
	#users = list(df.ix[:,'user_topic'])
	#print users
	if not os.path.exists(origin_fill):
		merge.get_tp_file(probs_file,df,origin_fill)
	merge.get_goal_file(probs_file,goal_fill,origin_fill)


def main():
	probs_file = result_path +'white_old.pro'
	df = result_path + 'svd0_origin.csv'
	origin_fill = result_path + 'white_old_svd0_origin.csv'
	goal_fill = result_path + 'white_old_svd0_goal_origin.csv'
	
	filter_twoparty_file(probs_file,df,origin_fill,goal_fill)
	
	
	#sample = sampling(probs_file,origin_fill)
	#print sample
	


if __name__ == '__main__':
	main()