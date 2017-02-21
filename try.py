#coding=utf-8


'''this script is to try the sampling method '''

import merge
import config
import pandas as pd


def get_Party(df,name):
	return df.ix[df['User_name'] == r]['Party'].tolist()[0]


if __name__ == '__main__':

	probs = merge.read_probs('./tmp')
	
	ic = {}
	for item in probs:
		ic[item[0]] = item[1]

	result = merge.accept_sampling_filter(probs,10)
	df = pd.read_csv(config.pro_data_path)
	
	for r in result:
		print r,ic[r],get_Party(df,r)
