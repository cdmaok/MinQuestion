# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import config
import sys
import re
import numpy as np
import math
from itertools import combinations

#groundtruth = config.groundtruth
groundtruth = '../../mq_data/groundtruth/trumper.gt'

def statistic(gt,model):
	
    #print gt,model
    temp1=gt      #A集合
    temp2=model    #B集合
    #print temp1,temp2
    tp=0
    fp=0
    fn=0
 
    for i in temp2:
        #print i
        if(i in temp1):#AB都有
            tp+=1
        else:#B有A没有
            fp+=1
    for i in temp1:
        if(i not in temp2):#A有B没有
            fn+=1
    #print tp,fp,fn
    if(tp==0):
        if(fp==0):
            precision=-1
        else:
            precision=0
        if(fn==0):
            recall=-1
        else:
            recall=0
        f=-1
    else:
        precision=1.0*tp/(tp+fp)
        recall=1.0*tp/(tp+fn)
        f=2*precision*recall/(precision+recall)#权重[0.5,0.5]
	
    #print '----',precision,recall,f
    return precision,recall,f



def get_ref(filename,candidate):
	f = open(filename)
	while True:
		line = f.readline().strip()
		if not line: break
		if line.startswith('<class'):
			tmp = line.split('.')
			fs = tmp[len(tmp)-1].replace('\'>','')
			#print fs
		if line.startswith('-------print feature'):
			line = f.readline().strip()
			querys = []
			while not line.startswith('['):
				querys.append(line)
				line = f.readline().strip()
			a,b,c = get_fields(querys)
			#print candidate,c
			p,r,f_m = statistic(candidate,c)
			print fs,p,r,f_m
			#print '\n'.join(c)
			#print 

def get_fields(array):
	indexs = []
	votes = []
	texts = []
	for q in array:
		fs = q.split('+')
		indexs.append(fs[0])
		votes.append(fs[1])
		texts.append(fs[2])
	return indexs,votes,texts

	
def get_can(groundtruth):
	
	f = open(groundtruth)
	candi_querys = []
	while True:
		line = f.readline().strip()
		if not line: break		
		candi_querys.append(line)
		
	return candi_querys
   
    
if __name__ == "__main__":
	print 'metric'
	if len(sys.argv) < 3:
		print 'need a filename'
		print 'python rouge.py groundtruth result.log'
		sys.exit()
	
	groundtruth = sys.argv[1]
	filename = sys.argv[2]
	print filename
	candidate = get_can(groundtruth)
	#print candidate
	get_ref(filename,candidate)