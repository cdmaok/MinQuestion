# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 09:57:48 2017

@author: nb
"""
import sys
sys.path.append("..")
import config
import sys
import re
import numpy as np
import math
from itertools import combinations

#groundtruth = config.groundtruth
groundtruth = '../../mq_data/groundtruth/white_old.gt'

def rouge(candidate,reference,rouge_type,n=1,Dskip=4):#n专门给ROUGE-N,L使用，Dskip专门给ROUGE-S使用
    if(rouge_type=='n'):#ROUGE-N
        if(type(candidate)==list):
            candidate=' '.join(candidate)
        tot_same=0
        tot_reference_len=0
        for one_reference in reference:
            r=one_reference.split(' ')
            c=candidate.split(' ')
            new_r=[]
            new_c=[]
            for i in range(len(r)-n+1):
                temp=' '.join(r[i:i+n])
                new_r.append(temp)
            for i in range(len(c)-n+1):
                temp=' '.join(c[i:i+n])
                new_c.append(temp) 
            same=0
            for i in new_r:
                if(i in new_c):
                    same+=1
            tot_same+=same
            tot_reference_len+=len(r)
        return 1.0*tot_same/tot_reference_len
    if(rouge_type=='l'):#ROUGE-L
        sum_l=0
        total_word_in_reference=0
        for r in reference:
            total_word_in_reference+=len(r.split(' '))
            sum_c=0
            for c in candidate:
                l= lcs_len(r,c,n)[0]
                sum_c+=l
            sum_l+=sum_c
        Rlcs=1.0*sum_l/total_word_in_reference
        return Rlcs
    if(rouge_type=='w'):#ROUGE-W(f(k)=k**2)
        sum_l=0
        total_word_in_reference=0
        for r in reference:
            total_word_in_reference+=len(r.split(' '))**2
            sum_c=0
            for c in candidate:
                l,p= lcs_len(r,c,n)
                sum_c+=l*p
            sum_l+=sum_c
        Rwlcs=math.sqrt(1.0*sum_l/total_word_in_reference)
        return Rwlcs
        
    if(rouge_type=='s'):#ROUGE-S
        r=''.join(reference)
        c=''.join(candidate)
        m=len(r.split(' '))
        n=len(c.split(' '))
        r_skip_bigram=get_skip_bigram(r,Dskip)
        c_skip_bigram=get_skip_bigram(c,Dskip)
        same=0
        for i in r_skip_bigram:
            if(i in c_skip_bigram):
                same+=1
        
        Rskip=1.0*same/(math.factorial(m)/(2*math.factorial(m-2)))
        Pskip=1.0*same/(math.factorial(n)/(2*math.factorial(n-2)))
        return Rskip,Pskip
        
        


def get_skip_bigram(summary,Dskip):#输入摘要，输出skip_bigram  
    r=summary.split(' ')
    skip_bigram=[]
    for i in combinations(range(len(r)),2):
        left=i[0]
        right=i[1]
        if(right-left-1<=Dskip):
            skip_bigram.append(r[left]+' '+r[right])
    return skip_bigram
        
    

def lcs_len(a,b,n):#返回LCS的长度
    '''
    a, b: strings
    '''
    new_a=[]
    new_b=[]
    a=a.split(' ')
    b=b.split(' ')
    for i in range(len(a)-n+1):
        temp=' '.join(a[i:i+n])
        new_a.append(temp)
    for i in range(len(b)-n+1):
        temp=' '.join(b[i:i+n])
        new_b.append(temp)
    a=new_a
    b=new_b
    n = len(a)
    m = len(b)
    
    l = [([0] * (m + 1)) for i in range(n + 1)]
    p=[([0] * (m + 1)) for i in range(n + 1)]
    direct = [([0] * m) for i in range(n)]#0 for top left, -1 for left, 1 for top
    
    for i in range(1,n + 1):
        for j in range(1,m + 1):
            if a[i - 1] == b[j - 1]:
                l[i][j] = l[i - 1][j - 1] + 1
                p[i][j]=p[i-1][j-1]+1
            elif l[i][j - 1] > l[i - 1][j]: 
                l[i][j] = l[i][j - 1]
                direct[i - 1][j - 1] = -1
            else:
                l[i][j] = l[i - 1][j]
                direct[i - 1][j - 1] = 1
    max_p=np.array(p).max()             
    return l[len(a)][len(b)],max_p

def get_lcs(direct, a, i, j):
    '''
    direct: martix of arrows
    a: the string regarded as row
    i: len(a) - 1, for initialization
    j: len(b) - 1, for initialization
    '''
    lcs = []
    get_lcs_inner(direct, a, i, j, lcs)
    return lcs
    
def get_lcs_inner(direct, a, i, j, lcs):    
    if i < 0 or j < 0:
        return
    
    if direct[i][j] == 0:
        get_lcs_inner(direct, a, i - 1, j - 1, lcs)
        lcs.append(a[i])
             
    elif direct[i][j] == 1:
        get_lcs_inner(direct, a, i - 1, j, lcs)
    else:
        get_lcs_inner(direct, a, i, j - 1, lcs)

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
			#print c
			print fs,rouge(candidate,c,'n',1),rouge(candidate,c,'l',1),rouge(candidate,c,'w',1),rouge(candidate,c,'s',1)
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
	print 'rouge'
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
	
	'''
	reference=['police killed the gunman', 'the gunman was shot down by police']
    candidate1=['police ended the gunman']
    #candidate2='the gunman murdered police'
    #print rouge(candidate1,reference,'n',1)



    reference=['police killed the gunman']
    candidate1=['the gunman murdered police']
    candidate2=['the gunman police killed']
    #print rouge(candidate1,reference,'l',1)
    
    reference=['police killed the gunman who injured 3 on campus']
    candidate1=['police killed the gunman and sealed off the scene']
    candidate2=['the police was killed and the gunman ran off']
    #print rouge(candidate2,reference,'w',1)
    
    reference=['police killed the gunman']
    candidate1=['the gunman kill police']
    candidate2=['the gunman police killed']
    candidate3=['police kill the gunman']
    #print rouge(candidate3,reference,'s')
    '''
    
        