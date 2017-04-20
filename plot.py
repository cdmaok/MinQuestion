#coding=utf-8
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import savefig
import pandas as pd
import matplotlib.colors as colors
import config
#result_path = config.result_path
pred = 'sim_tfidf'

def diff_color(filename,result_path):
    
    pre = []
    line_num = 0
    diff = {}
    list = []
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = eval(line)
            line_num += 1
            
            if line_num == 1:
				index = line.index(pred)			
				if pred == 'pre2':
					xname = line[index+8:]
					list = [index] + range(index+8,len(line))
				elif pred == 'pre3':
					xname = line[index+7:]
					list = [index] + range(index+7,len(line))					
				elif pred == '_pre2':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))
				elif pred == '_pre3':
					xname = line[index+1:]
					list = [index] + range(index+1,len(line))
				elif pred == 'sim_doc2vec':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))				
				x = [[] for i in xrange(len(xname))]
            else:
                if line[1] in diff:
                    diff[line[1]].append(line_num-2)
                else:
                    diff[line[1]] = [line_num-2]
                n = 0
                for i in list:
                    if i==index:
                        pre.append(line[i])
                    else:
                        x[n].append(line[i])
                        n = n + 1

    color = []
    for c in colors.cnames:
        color.append(c)
        if len(color) == len(diff):
            break
    #print(diff)


    mark = [".","v","<","s","p","h","8","d","1","2","3","4","o","*","+","^",">"]

    if len(color)*len(mark) < len(diff):
        print(len(color),len(diff))
        print("color is not enough")
    for i in xrange(len(xname)):
        k = 0
        for key in diff:
            tmp_x = []
            tmp_y = []
            for j in diff[key]:
                tmp_x.append(x[i][j])
                tmp_y.append(pre[j])
            plt.scatter(tmp_x,tmp_y,marker=mark[k%len(mark)],c=color[k%len(color)],label=xname[i])
            # plt.plot(tmp_x,tmp_y,'x',color[k],label=xname[i])
            k += 1
        plt.xlabel(xname[i])
        plt.ylabel('predictability')
        plt.xlim(0, max(x[i]))
        plt.ylim(0, max(pre))
        plt.title(xname[i]+'--'+'predictability')
        savefig(result_path + filename.split('/')[-1].replace('.log','_dsc_new_')  + xname[i])

def same_color(filename,result_path):
    pre = []
    line_num = 0  
    list = []	
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = eval(line)

            line_num += 1
            if line_num == 1:
				index = line.index(pred)			
				if pred == 'pre2':
					xname = line[index+8:]
					list = [index] + [i for i in range(index+8,len(line))]
				elif pred == 'pre3':
					xname = line[index+7:]
					list = [index] + range(index+7,len(line))					
				elif pred == '_pre2':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))
				elif pred == '_pre3':
					xname = line[index+1:]
					list = [index] + range(index+1,len(line))
				elif pred == 'sim_doc2vec':
					xname = line[index+3:]
					list = [index] + range(index+3,len(line))
				elif pred == 'sim_rake':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))					
				elif pred == 'sim_sen2vec':
					xname = line[index+4:]
					list = [index] + range(index+4,len(line))
				elif pred == 'sim_tfidf':
					xname = line[index+1:]
					list = [index] + range(index+1,len(line))
					
				x = [[] for i in xrange(len(xname))]
            else:
                n = 0
                for i in list:
#					if line[index-4] > 1 :				
					if i==index:
						pre.append(line[i])
					else:
						x[n].append(line[i])
						n = n + 1
	'''
    for i in xrange(len(xname)):
        plt.plot(x[i],pre,'r+',label=xname[i])
        plt.xlabel(xname[i])
        plt.ylabel('sim_doc2vec')
        plt.xlim(0, max(x[i]))
        plt.ylim(0, max(pre))
        plt.title(xname[i]+'--'+'sim_doc2vec')
        savefig(result_path + filename.split('/')[-1].replace('.log','_sc_')  + xname[i])		
   '''
    print xname

    for i in xrange(len(xname)):
        plt.plot(pre,x[i],'r+',label=xname[i])
        plt.xlabel(pred)
        plt.ylabel(xname[i])
        plt.xlim(0, max(pre))		
        plt.ylim(0, max(x[i]))
        print max(pre),max(x[i])
        plt.title(pred+'--'+xname[i])
        savefig(result_path + filename.split('/')[-1].replace('.log','_sc_')  + pred+'_'+xname[i])

		
def same_color_box(filename,result_path):
    pre = []
    #xname = ['sim_doc2vec','sim_rake','pos_response','response','nonresponse','coverage','gpos_response','gresponse','gnonresponse','gcoverage']
	
    #x = [[] for i in xrange(len(xname))]
    line_num = 0
    list = []
    count = 0
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            line = eval(line)
            line_num += 1

            if line_num == 1:

				index = line.index(pred)			
				if pred == 'pre2':
					xname = line[index+8:]
					list = [index] + range(index+8,len(line))
				elif pred == 'pre3':
					xname = line[index+7:]
					list = [index] + range(index+7,len(line))					
				elif pred == '_pre2':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))
				elif pred == '_pre3':
					xname = line[index+1:]
					list = [index] + range(index+1,len(line))
				elif pred == 'sim_doc2vec':
					xname = line[index+3:]
					list = [index] + range(index+3,len(line))
				elif pred == 'sim_rake':
					xname = line[index+2:]
					list = [index] + range(index+2,len(line))					
				elif pred == 'sim_sen2vec':
					xname = line[index+4:]
					list = [index] + range(index+4,len(line))
				elif pred == 'sim_tfidf':
					xname = line[index+1:]
					list = [index] + range(index+1,len(line))					
				x = [[] for i in xrange(len(xname))]
            else:
                n = 0
				
                for i in list:
					#if line[index] > 0 :
					#if line[index+3] > 0 :						
					if i==index:
						count = count + 1
						pre.append(line[i])
					else:
						x[n].append(line[i])
						n = n + 1

    print count
    num = 5
    #print xname
    for i in xrange(len(xname)):
        #print max(x[i]),min(x[i])
        #interval = 1.0*(max(x[i]) - min(x[i])) / num
        interval = 1.0*(max(pre) - min(pre)) / num
        rows = []
        for j in xrange(len(x[i])):
            #rows.append((pre[j],int((x[i][j]-0.0001-min(x[i]))/interval)))
            rows.append((x[i][j],int((pre[j]-0.0001-min(pre))/interval)))

        df = pd.DataFrame(rows,columns=['y','labels'])
        ax = df.boxplot(column='y',by='labels')

        label_name = []
        for j in xrange(num):
            label_name.append(round(interval * (j+1),2))
        ax.xaxis.set_ticklabels(label_name)

        plt.xlabel(pred)
        plt.ylabel(xname[i])
        plt.ylim(-0.1, max(x[i])+0.1)
        plt.title(pred+'--'+xname[i])
        savefig(result_path + filename.split('/')[-1].replace('.log','_b_')  + pred+'_'+xname[i])



def main():
    if len(sys.argv) < 4:
		print 'you need input 3 argv : 1.input file 2.output file path 3.method'
		print 'example: python plot.py ../mq_result/stat/plot/white_10.log ../mq_result/plot/white/box/ same_color_scatter'
		print 'method list : 1.same_color_scatter 2.diff_color_scatter 3.same_color_box\n'
		sys.exit()
    filename = sys.argv[1]
    result_path = sys.argv[2]   
    method = sys.argv[3]
	
    if(method == 'same_color_scatter'):
		same_color(filename,result_path)
    elif(method == 'diff_color_scatter'):
		diff_color(filename,result_path)
    elif(method == 'same_color_box'):
		same_color_box(filename,result_path)
    else: print('error input')

if __name__ == "__main__":
	main()