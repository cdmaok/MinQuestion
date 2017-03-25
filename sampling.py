# -*- coding: utf-8 -*-
import pandas as pd
import os
import merge
import config
from sklearn.utils import resample
import numpy as np
import numpy
data_path = config.data_path
result_path = config.result_path

def sampling(probs_file,origin_file):
	#sample_df = df.resample(frac=0.1)
	type = 0
	frac = 0.5
	merge.get_sample_file(probs_file,origin_file,frac,type)
	#return sample_df
	
def filter_twoparty_file(probs_file,df,origin_fill,goal_fill):
	df =  pd.read_csv(df,dtype={"user_topic":str,"Class":str})
	'''
	df = df.ix[:,[0,1,2,3,4,5,-1]]
	df = df.sample(frac=0.1,random_state = numpy.random.RandomState)
	users = list(df.ix[:,'user_topic'])
	print users
	'''
	if not os.path.exists(origin_fill):
		merge.get_tp_file(probs_file,df,origin_fill)
	merge.get_goal_file(probs_file,goal_fill,origin_fill)
	


def main():
	probs_file = result_path +'white_old.pro'
	df = result_path + 'knntext0_origin.csv'
	origin_fill = result_path + 'white_old_knntext0_origin.csv'
	goal_fill = result_path + 'white_old_knntext0_goal_origin.csv'
	#df = goal_fill
	#filter_twoparty_file(probs_file,df,origin_fill,goal_fill)
	origin_fill = goal_fill 
	#origin_fill = result_path + 'test/sonar.csv'
	for i in range(10):
		#print i
		frac = 0.5
		#sample = sampling(probs_file,origin_fill)
		sampled_df = merge.get_sample(probs_file,origin_fill,i,frac,0)
	#print sample
	
	


if __name__ == '__main__':
	main()