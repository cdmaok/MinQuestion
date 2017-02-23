# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 10:45:24 2016

@author: nb
"""

import pandas as pd
import re
import os


def traverseDir(route,file_type='xlsx'):
    path = os.path.expanduser(route)
    gt_file_name=[]
    for f in os.listdir(path):
        temp=re.findall(r'[A-Za-z\_0-9]+.'+file_type,f.strip())
        if(len(temp)!=0):
            gt_file_name.append(temp[0])
    return gt_file_name

def deal_with_gt(gt_filepath):
    gt=pd.DataFrame()
    for i in traverseDir(gt_filepath,'xlsx'):
        data=pd.read_excel(gt_filepath+i)
        gt=pd.concat([gt,data])
    gt=gt.dropna().drop_duplicates()
    return gt


def deal_with_model(model_filepath):
    model=pd.read_excel(model_filepath)
    model=model.ix[:len(model)-2,:]#去掉最后一行
    #if(len(re.findall(r'non_ensemble',model_filepath))==0):#是ensemble的
    #    model=model[model['#num']>=5]#投票次数大于等于5次的将被筛选出来
    return model


def statistic(gt,model):
    temp1=gt.index#A集合
    temp2=model.index#B集合
    tp=0
    fp=0
    fn=0
    for i in temp2:
        if(i in temp1):#AB都有
            tp+=1
        else:#B有A没有
            fp+=1
    for i in temp1:
        if(i not in temp2):#A有B没有
            fn+=1
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
    return precision,recall,f
    

if __name__ == '__main__':
 
    gt_file='G:\\Anaconda3\\code\\lab\\20170118_task_gt\\result\\whiteold\\selected\\'
    gt=deal_with_gt(gt_file)
    model_package='G:\\Anaconda3\\code\\lab\\0125\\'
    whole_result=[]
    for i in ['ensemble\\','ensemble_fill\\','non_ensemble\\']:
        path1=model_package+i
        df=pd.DataFrame(columns=['type','precision','recall','f'])
        for j in traverseDir(path1):
            path2=path1+j
            model=deal_with_model(path2)
            stat=statistic(gt,model)
            df.loc[len(df)]=[j,stat[0],stat[1],stat[2]]
        whole_result.append(df)
        df.to_excel(model_package+'result\\'+i+'result.xlsx')
            
    
    
    
        
    