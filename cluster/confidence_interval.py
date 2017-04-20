# -*- coding: utf-8 -*-

def get_top_q(_list):
	
	cov = [[] for i in _list]
	max = 0	
	max_v = 0
	
	
	for j in range(len(_list)):
		cov[j].append(_list[j])
		print _list[j]
		q1_yes = _list[j][3][4]
		q1_po = _list[j][4]
		q1_ci = _list[j][4]
		print q1_yes,q1_ci
		q1_l = q1_yes - q1_ci
		q1_r = q1_yes + q1_ci
		for k in range(len(_list)):
			if not j == k:
				q2_po = _list[k][4]
				
				if q1_po == q2_po:
					q2_yes = _list[k][3][4]
				else: q2_yes = _list[k][3][5]
				
				q2_ci = _list[k][4]
				q2_l = q2_yes - q2_ci
				q2_r = q2_yes + q2_ci
				
				if q2_l > q1_l and q2_r < q1_r:
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
	
def coverage():
	
	filename = './ci_test'
	dict = get_dict_c(filename)
	
	
	f = 0
	for key in dict:
		q = dict[key]
		#print q
		
		_list = q

		lq = []
				
		while not len(_list) == 0 :
			#print f
			f = f + 1
			tq,_list =get_top_q(_list)
			lq.append(tq)
		
		#print lq
		print '\n'
		for i in lq:
			print i


def read_data(filename):
	probs = [eval(e.strip()) for e in open(filename).readlines() ]
	return probs


def confident_interval():
	filename = './aptc2_po'
	query = read_data(filename)
	
	for i in query:
		vote = i[3][2]
		ci = 1/math.sqrt(vote)
		i.append(ci)
		print i

		
def main():
	#confident_interval()
	coverage()



if __name__ == '__main__':
	main()