#coding=utf-8


'''this script is to try the sampling method '''

import merge




if __name__ == '__main__':

	probs = merge.read_probs('./tmp')
	
	ic = {}
	for item in probs:
		ic[item[0]] = item[1]

	result = merge.accept_sampling(probs,10)
	for r in result:
		print r,ic[r]
	
