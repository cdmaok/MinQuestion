#coding=utf-8
import query_match as np
import pandas as pd
import sys
import os
sys.path.append("..")
import config
result_path = config.result_path

def generate_match(gt_yes,gt_no):
    output = sys.stdout
    outputfile = open(result_path + 'stat/new_matrix/data/query_match.txt','w')
    sys.stdout = outputfile
    print "(i,j,q_1,q_2,polarity)"
    for iter,yes in enumerate(gt_yes):
            for i in range(len(yes)):
                for j in range(i+1,len(yes)):
                    print[iter,yes[i][0],yes[j][0],yes[i][1],yes[j][1],1]

    for iter,no in enumerate(gt_no):
             for i in range(len(no)):
                 for j in range(i+1,len(no)):
                     print[iter,no[i][0],no[j][0],no[i][1],no[j][1],1]

    for iter,yes in enumerate(gt_yes):
        for i in range(len(yes)):
            for j in range(len(gt_no[iter])):
                print[iter,yes[i][0],gt_no[iter][j][0],yes[i][1],gt_no[iter][j][1],0]
    outputfile.close()
    sys.stdout = output



def read_data(filename):
	data = [eval(e.strip()) for e in open(filename).readlines() ]
	return data


def main():
    gt_yes = read_data(result_path + 'stat/new_matrix/data/gt_yes_0412')
    gt_no = read_data(result_path + 'stat/new_matrix/data/gt_no_0412')
    generate_match(gt_yes,gt_no)

if __name__ == '__main__':
    main()

