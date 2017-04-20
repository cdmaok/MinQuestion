# -*- coding: utf-8 -*-
import numpy as np
import option
import sys
sys.path.append("..")
import query_user_comment
import os
import math
Path = './cluster/'

def read_data(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs
	
def get_cl(cluster):
	cl = [eval(e.strip()) for e in open(cluster).readlines() if e.startswith('[') ]
	cli = []
	for a in cl:
		cli.append(a[1])

	return cl,cli

def get_cluster_vote():
	f_query_pair = '/home/yangying/mq_result/stat/new_matrix/query_vpro_all'
	query_pair = read_data(f_query_pair)
	
	
	c = ['ap_pm','ap_me2','ap_tcomment','ap_tcomment_m2','ap_tcomment_m3','tcs_km_10','tcs_km_121']
	f_cluster = Path + c[3]
	cluster,cluster_list = get_cl(f_cluster)
	
	for i in cluster:
		c = i
		
		q = query_pair[c[1]]
		
		c.append(q[1])
		
		print c
	
def get_dict_c(filename):
	c = read_data(filename)
	dict = {}
	
	for i in c:
		if i[0] in dict.keys():
			dict[i[0]].append(i)
		else:
			dict[i[0]] = [i]
			
	#print dict
	return dict

def get_comment(_list,c_file):
	
	f = open(c_file,'w')
	for j in range(len(_list)):
		for k in range(len(_list)):
			if j < k:
				pair = [_list[j][0],_list[j][1],_list[k][1],_list[j][2],_list[k][2],'?']				
				#print pair
				f.write(str(pair)+'\n')
	f.close()			
		
	
def get_top_q(_list,c_file):
	
	cov = [[] for i in _list]
	max = 0
	e = 0.1
	max_v = 0
	
	
	for j in range(len(_list)):
		cov[j].append(_list[j])
		for k in range(len(_list)):
			if not j == k:
				if j < k:
					q1,q2,q3,q4 = [_list[j][1],_list[k][1],_list[j][2],_list[k][2]]
				else:
					q1,q2,q3,q4 = [_list[k][1],_list[j][1],_list[k][2],_list[j][2]]
				
				p_y1,p_y2,p_same,p_cont = option.optionAPI(q1,q2,q3,q4,c_file)
				
				tmp = _list[k][3][4]
				if p_same < p_cont:
					tmp = _list[k][3][5]
					
				#print p_y1,p_y2,p_same,p_cont,tmp
				real_e = abs(_list[j][3][4]-tmp)
				if real_e < e:
					cov[j].append(_list[k])
	
		l = len(cov[j])
		#print cov[j]
		if l > max:
			max = l
			max_i = j
		elif l == max and cov[j][0][3][2] > max_v:  #t = sorted(range(len(h_i)),key=lambda k:h_i[k],reverse=False)
			max_v = cov[j][0][3][2]
			max = l
			max_i = j
			
		
	m = _list[max_i]		
	
	#print cov
	#print _list, cov[max_i]
	for i in cov[max_i]:
		#print i
		_list.remove(i)
	#print 'final----',max,max_i,cov[max_i][0][0],_list		
	return m,_list	

def get_top_q2(_list,c_file):
	
	cov = [[] for i in _list]
	max = 0
	e = 0.1
	max_v = 0
	
	
	for j in range(len(_list)):
		cov[j].append(_list[j])
		for k in range(len(_list)):
			if not j == k:
								
				j_po = _list[j][4]
				k_po = _list[k][4]
				
				if j_po == k_po:
					tmp = _list[k][3][4]
				else: tmp = _list[k][3][5]
					
				#print p_y1,p_y2,p_same,p_cont,tmp
				real_e = abs(_list[j][3][4]-tmp)
				if real_e < e:
					cov[j].append(_list[k])
	
		l = len(cov[j])
		#print cov[j]
		if l > max:
			max = l
			max_i = j
		elif l == max and cov[j][0][3][2] > max_v:  #t = sorted(range(len(h_i)),key=lambda k:h_i[k],reverse=False)
			max_v = cov[j][0][3][2]
			max = l
			max_i = j
			
		
	m = _list[max_i]		
	
	#print cov
	#print _list, cov[max_i]
	for i in cov[max_i]:
		#print i
		_list.remove(i)
	#print 'final----',max,max_i,cov[max_i][0][0],_list		
	return m,_list
	
def compute_option():
	filename = './result_cluster_vote/cluster_vote_aptc2'
	dict = get_dict_c(filename)
	c_file = '/home/yangying/MinQuestion/cluster/result_cluster_vote/pair_o'
	part_comment = '/home/yangying/MinQuestion/cluster/result_cluster_vote/comment_o'
	group = '/home/yangying/mq_result/stat/new_matrix/pro/dem.pro'
	e = 0.1
	f = 0
	'''
	
	for key in dict:
		q = dict[key]
		
		f = open(c_file,'a')
		
		j = 0
		for k in range(1,len(q)):
			pair = [q[j][0],q[j][1],q[k][1],q[j][2],q[k][2],'?']				
			#print pair
			f.write(str(pair)+'\n')
		f.close()
		
	query_user_comment.get_part_comment(group,c_file,part_comment)
	'''	
	for key in dict:
		q = dict[key]
		p_same,p_cont = [0,0]
		j = 0
		j_po = '1'
		q[j].append(j_po)
		print q[j]
		for k in range(1,len(q)):
			k_po = '1'
			q1,q2,q3,q4 = [q[j][1],q[k][1],q[j][2],q[k][2]]							
			p_y1,p_y2,p_same,p_cont = option.optionAPI(q1,q2,q3,q4,c_file)
						
			if p_same < p_cont:
				k_po = '0'
			
			q[k].append(k_po)
			print q[k]
			
def confident_interval():
	filename = './aptc2_po'
	query = read_data(filename)
	
	for i in query:
		vote = i[3][2]
		ci = 1/math.sqrt(vote)
		i.append(ci)
		print i
	
	
	
		
	
def compute_cov():
		
	filename = './result_cluster_vote/cluster_vote_aptc2'
	dict = get_dict_c(filename)
	c_file = '/home/yangying/MinQuestion/cluster/result_cluster_vote/pair'
	part_comment = '/home/yangying/MinQuestion/cluster/result_cluster_vote/comment'
	group = '/home/yangying/mq_result/stat/new_matrix/pro/dem.pro'
	e = 0.1
	f = 0
	for key in dict:
		q = dict[key]
		#print q
		
		_list = q

		lq = []
		#if not os.path.exists(c_file):
		get_comment(_list,c_file)
		#if not os.path.exists(part_comment):
		query_user_comment.get_part_comment(group,c_file,part_comment)
		
		while not len(_list) == 0 :
			#print f
			f = f + 1
			tq,_list =get_top_q(_list,c_file)
			lq.append(tq)
		
		#print lq
		print '\n'
		for i in lq:
			print i


	
	
def main():
	#get_cluster_vote()
	#compute_cov()
	#compute_option()
	confident_interval()
	#a = read_data('./user')
	#print a[0]
		
	
	

if __name__ == '__main__':
	main()
