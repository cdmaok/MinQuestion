from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from PyDictionary import PyDictionary
from nltk.stem.porter import PorterStemmer

dictionary=PyDictionary()



def test():
	print (dictionary.synonym("Life"))
	print (dictionary.antonym("Life"))

def test2():
	for i in wn.all_synsets():
		if i.pos() in ['a', 's']: # If synset is adj or satelite-adj.
			for j in i.lemmas(): # Iterating through lemmas for each synset.
				if j.antonyms(): # If adj has antonym.
					# Prints the adj-antonym pair.
					print j.name(), j.antonyms()[0].name()

					
def process(q1,q2):
	porter_stemmer = PorterStemmer()
	q1 = q1.decode('gbk','ignore')
	q2 = q2.decode('gbk','ignore')
	q1 = word_tokenize(q1.lower())
	q2 = word_tokenize(q2.lower())
	q1 = words = [w for w in q1 if w.isalpha()]
	q2 = words = [w for w in q2 if w.isalpha()]
	w1 = set([w for w in q1])
	w2 = set([w for w in q2])
	w1 =w1 | set([porter_stemmer.stem(w)  for w in q1])
	w2 =w2 |  set([porter_stemmer.stem(w) for w in q2])

	#un = w1 & w2
	#w1 = w1 - un
	#w2 = w2 - un
	return w1,w2
	
def read_words(filename):
	#filename = './negative-words.txt'
	words = set([e.strip() for e in open(filename).readlines() if not e.startswith(';')])
	#print words
	return words
	
	
def negative(q1,q2):

	neg = read_words('./negative-words.txt')
	pos = read_words('./positive-words.txt')
	#neg = set(['illegal','right','out','against','bad','evil','outlaw','repeal','oppose','abolish','not','rid','weak','mediocre','failure','stop','remove','unfit','embarrass','terrible','leave','reject'])
	w1,w2 = process(q1,q2)
	#print w1,w2
	f1 =  w1&neg
	f2 =  w2&neg
	f3 =  w1&pos
	f4 =  w2&pos
	if 'or' in w1:
		f1 = []
		f3 = ['good']
	if 'or' in w2:
		f2 = []
		f4 = ['good']
		
	p_same,p_cont = [0,0]
	'''
	if len(f1)==0 and len(f2)==0: 
		print 'a' 
		p_same,p_cont = antonyms(w1,w2)
	elif abs(len(f1)-len(f2))%2 == 1:
		print 'b' 
		print q1,q2
		p_same,p_cont = [0,1]
	else: 
		print 'c' 
		p_same,p_cont = [1,0]
	'''
	a =abs(len(f1)-len(f2))
	
	#print f1,f2
	if len(f3)>0 and len(f4)>0:
		#print q1,q2
		p_same,p_cont = [1,0]
	
	
	if a%2 == 1:
		#print f1,f2
		p_same,p_cont = [0,1]
	elif len(f1)>0 and len(f2)>0:
		p_same,p_cont = [1,0]
	
	
	
	return p_same,p_cont
	
	
					
def antonyms(q1,q2):
	w1,w2 = process(q1,q2)
	#w1,w2 = [q1,q2]
	print w1,w2
	#flag = False
	p_same,p_cont = [0,0]
	q1_l = []
	q1_a = []
	
	for i in w1:
		#print i
		#print (dictionary.antonym(i))
		
		tmp = wn.synsets(i)
		for j in tmp:
			le = j.lemmas()
			for k in le:
				q1_l.append(k.name())
				an = k.antonyms()
				if not an == []:
					#print an
					for a in an:
						q1_a.append(a.name())

						
	q2_l = []
	q2_a = []
	for i in w2:
		#print i
		#print (dictionary.antonym(i))
		
		tmp = wn.synsets(i)
		for j in tmp:
			le = j.lemmas()
			for k in le:
				q2_l.append(k.name())
				an = k.antonyms()
				if not an == []:
					#print an
					for a in an:
						q2_a.append(a.name())	
	
	#print set(q1_l),'\n',set(q1_a)
	#print set(q2_l),'\n',set(q2_a)
	
	if set(q1_a)&set(q2_l) or  set(q1_l)&set(q2_a):
		#print 'different',set(q1_a)&set(q2_l),'or',set(q1_l)&set(q2_a)
		p_same,p_cont = [0,1]
		return p_same,p_cont
	
	if set(q1_a)&set(q2_a) or  set(q1_l)&set(q2_l):
		#print 'same,',set(q1_a)&set(q2_a),'or',set(q1_l)&set(q2_l)
		p_same,p_cont = [1,0]
	
	
	
	print '-----',p_same,p_cont
	return p_same,p_cont
					
def main():
	q1 = 'Do we really deserve to have Donald Trump as our president?  '
	q2 = 'Should Donald Trump be elected as president?  '
	negative(q1,q2)
	#antonyms(q1,q2)
	#read_words()
	#test()

if __name__=='__main__':
	main()