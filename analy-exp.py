#coding=utf-8

### this script is to analyse the exp data
import config
import sys



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


def read_file(filename):
	f = open(filename)
	while True:
		line = f.readline().strip()
		if not line: break
		if line.startswith('-------print feature'):
			line = f.readline().strip()
			querys = []
			while not line.startswith('['):
				querys.append(line)
				line = f.readline().strip()
			a,b,c = get_fields(querys)
			print '\n'.join(c)
			print 




if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'need a filename'
		sys.exit()
	filename = sys.argv[1]
	read_file(filename)
