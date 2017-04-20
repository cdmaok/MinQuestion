# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import ast
import os
import sys
sys.path.append("..")
import query_user_comment
import numpy
from nltk.corpus import wordnet as wn
import antonyms

def anotonyms(q1,q2,vote_file,dict_pro,comment_file,q3,q4):
		
	p_same,p_cont = antonyms.negative(q3,q4)

		
	return 0,0,p_same,p_cont			


def combine(q1,q2,vote_file,dict_pro,comment_file,q3,q4):
	
	#print 'combine'
	p_y1,p_y2,p1_same,p1_cont = comment(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
	p_y1,p_y2,p2_same,p2_cont = vote(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
	p_y1,p_y2,p3_same,p3_cont = group_vote(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
	p_y1,p_y2,p4_same,p4_cont = anotonyms(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
	
	#a,b,c,d = [1.7168810998741986, 4.5790304413645311, 0.83801622991580382, 3.7983253894340923] #0.62
	#a,b,c,d = [0.0010601556924015839, 2.0061781834136059, 0.00098805649817846979, 2.0029266790644753]
	#a,b,c,d = [0, 0, 0, -2]
	
	a,a2,b,b2,c,c2,d,d2 = [0.0035266975321910647, -0.0023656320480996912, 0.035766547840436362, -2.0078039355881687, 0.0069513917816627253, 0.00084909237847118806, 0.0016257521745646677, -2.0061781834136099]
	
	#a,a2,b,b2,c,c2,d,d2 = [0.6189220927209732, -0.78738889054068228, 1.1770208143029015, -1.9846152629555422, 1.4483110846775102, -0.28645453153542549, 0.69113432459755897, -3.9008371192007538]
	
	#print '======final',p1_same,p1_cont,p2_same,p2_cont,p3_same,p3_cont
	p1 = p1_same - p1_cont
	p2 = p2_same - p2_cont
	p3 = p3_same - p3_cont
	p4 = p4_same - p4_cont
	
		
	#p_same, p_cont = a * numpy.array([p1_same,p1_cont]) + b * numpy.array([p2_same,p2_cont]) + c * numpy.array([p3_same,p3_cont]) +  d * numpy.array([p4_same,p4_cont])
	
	p_same = a * p1_same + b * p2_same + c * p3_same +  d * p4_same
	p_cont = a2 * p1_cont + b2 * p2_cont + c2 * p3_cont +  d2 * p4_cont
	p_cont = -p_cont
	
	return [p1_same,p1_cont,p2_same,p2_cont,p3_same,p3_cont,p4_same,p4_cont],[p1,p2,p3,p4],p_same,p_cont


def group_vote_2(q1,q2,vote_file,dict_pro,comment_file):
	
	q1_y = 0 
	q1_n = 0
	q2_y = 0
	q2_n = 0
	group1 = 0
	group2 = 0
	dict1 = get_q_dict(q1,vote_file)
	dict2 = get_q_dict(q2,vote_file)
	a = 1	
	for key in dict1:
		if key in dict_pro:
			if dict_pro[key] == 1 :
				group1 = group1 + 1
				if dict1[key] == 'yes':
					q1_y = q1_y + 1
				elif dict1[key] == 'no':
					q1_n = q1_n + 1
		else: print 'user not in pro'
	
	for key in dict2:
		if key in dict_pro:
			if dict_pro[key] == 1 :
				group2 = group2 + 1
				if dict2[key] == 'yes':
					q2_y = q2_y + 1
				elif dict2[key] == 'no':
					q2_n = q2_n + 1
		else: print 'user not in pro'
	
	q1_vote = q1_y + q1_n
	q2_vote = q2_y + q2_n
	
	#print q1_vote == group1
	#print q2_vote == group2
	
	if not q1_vote == 0:		
		p_y1 = q1_y / float(q1_vote)
		p_n1 = q1_n / float(q1_vote)
	else:
		p_y1 = 0
		p_n1 = 0
	
	if not q2_vote == 0:		
		p_y2 = q2_y / float(q2_vote)
		p_n2 = q2_n / float(q2_vote)
	else:
		p_y2 = 0
		p_n2 = 0


	p_same = 1 - cosine([p_y1, p_n1], [p_y2, p_n2])
	p_cont = 1 - cosine([p_y1, p_n1], [p_n2, p_y2])
	
	#print '-----------------------------q1-yes=',p_y1 , 'q2-yes=',p_y2, 'pro_same=',p_same,'pro_cont=',p_cont
	return p_y1,p_y2,p_same,p_cont		

def get_g_dict(q1,vote_file):
	v = vote_file[q1]
	dict = {}
	for i in range(1,len(v)):
		#print i
		i = v[i]
		#print i
		#if i[2] == '0': continue

		if not i[2] in dict:
			dict[i[2]] = [i[1]]
		else: 
			dict[i[2]].append(i[1])

	#print dict
	#print dict.keys()
	return dict	
	

	
	
def group_vote(q1,q2,vote_file,dict_pro,comment_file,q3,q4):
	
	dict1 = get_g_dict(q1,vote_file)
	dict2 = get_g_dict(q2,vote_file)
	dict3={}
	dict4={}
	union = set(dict1.keys())|set(dict2.keys())
	inter = set(dict1.keys())&set(dict2.keys())
	#print inter
	n_group = len(inter)
	p_same_total, p_cont_total = [0,0]
	for key in union:
		#print '--',key
		#if key == '0': continue
			
		if key in dict1:			
			p_y1 = dict1[key].count('yes')/ float(len(dict1[key]))
			p_n1 = dict1[key].count('no')/ float(len(dict1[key]))
		else:
			p_y1 = 0
			p_n1 = 0
		
		if key in dict2:			
			p_y2 = dict2[key].count('yes')/ float(len(dict2[key]))
			p_n2 = dict2[key].count('no')/ float(len(dict2[key]))
		else:
			p_y2 = 0
			p_n2 = 0
			
		#print '1-',p_y1,'2-',p_n1,'3-',p_y2,'4-',p_n2
		if not ([p_y1, p_n1]==[0,0] or [p_y2, p_n2]==[0,0]):
			
			p_same = 1 - cosine([p_y1, p_n1], [p_y2, p_n2])
			p_cont = 1 - cosine([p_y1, p_n1], [p_n2, p_y2])
		else:
			p_same = 0 
			p_cont = 0
		
		dict3[key] = [p_y1, p_n1]
		dict4[key] = [p_y2, p_n2]
		#print p_same,p_cont
		p_same_total, p_cont_total = numpy.array([p_same_total, p_cont_total]) + numpy.array([p_same,p_cont])
	
	if not n_group == 0:
		a = p_same_total/float(n_group)
		b = p_cont_total/float(n_group)
	else:
		#print 'not intersection'
		a,b = [0,0]
		
	#print '-----',dict3,dict4,a, b
	return dict3,dict4,a, b
	#return dict3,dict4,p_same_total, p_cont_total

		
		
	

def get_pro_dict(profile):
	pros = read_probs(profile)	
	dict = {}	
	for i in pros:
		#print i
		if not i[0] in dict:
			dict[i[0]] = i[1]
		else: 
			print 'dict error'
			break
	
	#print dict
	return dict



def read_probs(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs

def read_dic(filename):
    line_num = 0
    with open(filename) as f:
      for line in f:
            line_num += 1
            if line_num != 1:
              dict = line
              break
    dict = ast.literal_eval(dict)
    return dict
	
def comment(q1,q2,vote_file,dict_pro,dict,q3,q4):
	
	key = (q1,q2)
	co = dict[key]
	yy,nn,yn,ny = co[2:]
	p_same = yy + nn
	p_cont = yn + ny
	return (yy,nn),(yn,ny),p_same,p_cont

def get_q_dict(q1,vote_file):
	v = vote_file[q1]
	dict = {}
	for i in range(1,len(v)):
		#print i
		i = v[i]
		#print i
		if not i[0] in dict:
			dict[i[0]] = i[1]
		else: 
			print 'dict error'
			break
	
	#print dict
	return dict
	
	
def vote(q1,q2,vote_file,dict_pro,comment_file,q3,q4):
	dict1 = get_q_dict(q1,vote_file)
	dict2 = get_q_dict(q2,vote_file)
	
	intsec = 0
	same = 0
	for key in dict1:
		if key in dict2:
			intsec = intsec + 1
			if dict1[key] == dict2[key]:
				same = same + 1
	
	#print 'intsec = ', intsec, ' same = ',same			
	if not intsec == 0 :
		p_same = same / float(intsec)
		p_cont = 1 - p_same
	else:
		p_same = 0
		p_cont = 0
		
	return intsec,same,p_same,p_cont

def all_same(q1,q2,vote_file,dict_pro,comment_file,q3,q4):
	return 1,1,1,0

def get_cl(cluster):
	cl = [eval(e.strip()) for e in open(cluster).readlines() if e.startswith('[') ]
	cli = []
	for a in cl:
		cli.append(a[1])

	return cl,cli

def optionAPI(q1,q2,q3,q4,c_file,m=5):
	
	#print 'API'
	#f_query_pair = c_file
	
	c = ['ap_pm','ap_me2','ap_tcomment','ap_tcomment_m2','ap_tcomment_m3','tcs_km_10','tcs_km_121']
	f_cluster = c[3]
	
	#f_vote = '/home/yangying/mq_result/stat/new_matrix/query_user_vote'
	#f_vote = '../preprocess/info'
	f_vote = '../preprocess/data/group2'

	part_comment = '/home/yangying/MinQuestion/cluster/result_cluster_vote/comment_o'
	group = '/home/yangying/mq_result/stat/new_matrix/pro/dem.pro'
	profile = group

	method_list = [ all_same, comment, vote, group_vote, anotonyms, combine]
	method_list_name = [ 'all_same', 'comment', 'vote', 'group_vote', 'anotonyms', 'combine']
	method = method_list[m]
	#print '\nmethod:',method_list_name[m],'  cluster_file:',f_cluster,'\n\nerrorlog:'
	vote_file = ''
	dict_pro = ''
	comment_file = ''	
	
	if m == 1 or m == 5:
		#if not os.path.exists(part_comment):
			#query_user_comment.get_part_comment(group,f_query_pair,part_comment)

		comment_file = read_dic(part_comment)
			
	if m == 2 or m == 5:
		vote_file = read_probs(f_vote)
	if m == 3 or m == 5:
		vote_file = read_probs(f_vote)
		dict_pro = get_pro_dict(profile)
		
	#p_same = 0
	#p_cont = 0
	#print method
	
	p_y1,p_y2,p_same,p_cont = method(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
	#print p_y1,p_y2,p_same,p_cont
	return p_y1,p_y2,p_same,p_cont
			

	
	
	
def predict_option(m = 1,cn=3):
	
	f_query_pair = '/home/yangying/mq_result/stat/new_matrix/data/query_match.txt'
	query_pair = read_probs(f_query_pair)
	c = ['ap_pm','ap_me2','ap_tcomment','ap_tcomment_m2','ap_tcomment_m3','tcs_km_10','tcs_km_121']
	f_cluster = './cluster/' + c[cn]
	cluster,cluster_list = get_cl(f_cluster)
	
	#f_vote = '/home/yangying/mq_result/stat/new_matrix/query_user_vote'
	#f_vote = '../preprocess/info'
	f_vote = '../preprocess/data/group2'
	#profile = '/home/yangying/mq_result/stat/white.pro'
	profile = '/home/yangying/mq_result/stat/new_matrix/pro/dem.pro'
	part_comment = './tfidf_alluser_query_match'
	group = profile
	#group = 'stat/alluser.pro'
	
	query_pair_in = 0
	query_pair_out = 0
	n_true = 0
	n_flase = 0
	accuracy = 0
	method_list = [ all_same, comment, vote, group_vote, anotonyms, combine]
	method_list_name = [ 'all_same', 'comment', 'vote', 'group_vote', 'anotonyms', 'combine']
	method = method_list[m]
	#print '\nmethod:',method_list_name[m],'  cluster_file:',f_cluster,'\n\nerrorlog:'
	vote_file = ''
	dict_pro = ''
	comment_file = ''	
	
	if m == 1 or m == 5:
		if not os.path.exists(part_comment):
			query_user_comment.get_part_comment(group,f_query_pair,part_comment)

		comment_file = read_dic(part_comment)
			
	if m == 2 or m == 5:
		vote_file = read_probs(f_vote)
	if m == 3 or m == 5:
		vote_file = read_probs(f_vote)
		dict_pro = get_pro_dict(profile)
	

	
	for q in query_pair:
		q1 = q[1]
		q2 = q[2]
		true_pola = q[5]
		#print q1,q2,true_pola
		c_q1 = cluster[cluster_list.index(q1)]
		c_q2 = cluster[cluster_list.index(q2)]
		#print c_q1,c_q2
		p_same = 0
		p_cont = 0
		polarity = '?'
		#polarity = 1
		q3,q4 = ['','']
		if c_q1[0] == c_q2[0]:
			query_pair_in = query_pair_in + 1
			if m == 5 or  m == 4: q3,q4 = [q[3],q[4]]
			p_y1,p_y2,p_same,p_cont = method(q1,q2,vote_file,dict_pro,comment_file,q3,q4)
			
			'''
			a = p_y1+ [true_pola]
			#x.append(a)
			output = sys.stdout
			outputfile = open('/home/yangying/MinQuestion/cluster/tmp_p','a')
			sys.stdout = outputfile
			print a
			#outputfile.write(str[])
			outputfile.close()
			sys.stdout = output
			'''
			

			if 	p_same < p_cont :
				polarity = 0
			if 	p_same > p_cont :
				polarity = 1

				
			if polarity == true_pola:
			#if polarity == true_pola or true_pola == 1:
				#print '-----------------------------r1=',p_y1 , 'r2=',p_y2, 'pro_same=',p_same,'pro_cont=',p_cont
				#print 'bingo',c_q1,c_q2,true_pola,polarity
				n_true = n_true + 1
			else: 
				n_flase = n_flase + 1
				#print '-----------------------------r1=',p_y1 , 'r2=',p_y2, 'pro_same=',p_same,'pro_cont=',p_cont
				#print '!!!Predict error',c_q1,c_q2,true_pola,polarity
			
			
		else: 
			#print '*** This two query are not in a cluster',c_q1,c_q2
			query_pair_out = query_pair_out + 1
	
	#print '  cluster_file:',f_cluster,
	#print '\n\n',['n_true','n_flase','query_pair_in','query_pair_out','accuracy']
	if not (query_pair_in + query_pair_out ) == len(query_pair):
		print 'Error '
	else:	
		accuracy = n_true / float(len(query_pair))	
		print [n_true,n_flase,query_pair_in,query_pair_out,accuracy]
	
	

def main():
	m = 5
	#predict_option(m)
	
	for cn in range(7):
		print cn
		for m in range(6):
			predict_option(m,cn)
	

if __name__=='__main__':
	main()