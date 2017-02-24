# -*- coding: utf-8 -*-
import numpy as np
import types
import pandas as pd
from copy import copy
import  threading

class DecisionTree(threading.Thread):
    
    def __init__(self,sampled_df,f_num):
        threading.Thread.__init__(self)
        self.sampled_df = sampled_df
        self.topics = []
        self.num = f_num
        self.train=np.array([])
        self.labels=[]
        self.labelsT=[]
        self.Test=np.array([])
        self.usedThresholds={}
        for i in range(0,6777):
            self.usedThresholds[i]=set()
        self.Tree=np.ones((10000,1))
        self.Thresholds=np.ones((10000,1))
        self.decisions={}
        self.Tree=-1*self.Tree
        self.last=0
        pass
    
    def run(self):
        self.loadTrain(self.sampled_df)                
    
    def loadTrain(self,df):
        
        #f=open('horseTrain.txt')
        #csvname = '~/mq_result/white_old_knntext0_goal_origin.csv'
        #df = pd.read_csv(csvname,dtype={"user_topic":str,"Class":str})
        #df.describe()
        topic_nums = len(df.iloc[0])-1
        self.train,self.labels = self.getXY(df)
        feature = []		       
        #print 'train set is ',self.train.shape,type(self.train)
        #print 'labels set is ',self.labels.shape,type(self.labels)
        #print 'Now calling contructTree'
        
        self.topics = self.contructTree(self.train,1,self.labels)
        #print self.topics

    def getXY(self,df):
        def replaceLabel(x):
            x = int(x)
            tmp = 1 if x == 4 else -1
            #tmp = 1 if x == 1 else -1
            return tmp
        headers = list(df.columns)
        start = headers.index('user_topic')
        end = headers.index('Class')
        x = df.ix[:,start + 1:end].as_matrix()
        y = df.ix[:,end].apply(replaceLabel).as_matrix()
        return x,y            
        
    def contructTree(self,data1,nodeNum,labels1):
            #since its a recursive function we need to have a base case. return when the number of wrong classes is 0. maybe 
            #we can chane it later
            #print 'nodeNum is ',nodeNum
            data=copy(data1)
            labels=copy(labels1)
            #print labels
            rows=np.where(labels== 1 )[0]
            rows2=np.where(labels== -1 )[0]
            #print 'number of Rep in this node is ',rows.shape[0]
            #print 'number of Dem in this node is ',rows2.shape[0]
            if rows.shape[0]==0 or rows2.shape[0]==0:
                self.decisions[nodeNum]=(rows.shape[0],rows2.shape[0])
                return
            
            IGA=[]
            thresholds=[]
            t = []
            fea = []
            attr_num = data.shape[1]
            #print attr_num
            #mutex = threading.Lock()
            #mutex.acquire([timeout])
            for attr in range(0,attr_num):
                #print '-----------------no. ',attr	
                				
                thresh,IG=self.findThresholdAndIG(data,attr,labels)
                #print thresh,IG
                IGA.append(IG)
                thresholds.append(thresh)
            #mutex.release()                
            #print IGA
            #print thresholds
						
            t = sorted(range(len(IGA)),key=lambda k:IGA[k],reverse=True)
            #print '---feature-',t[:10]
            test = [IGA[i] for i in t[:10]]
            #print '----entropy---',test[:10]
            th = [thresholds[i] for i in t[:10]]
            #print '----threshold---',th[:10]
            return t[:10]
			
    def findThresholdAndIG(self,data1,Attr,labels1):
        #print 'trying attribute ',Attr
        data=copy(data1)
        labels=copy(labels1)
        values=set(data[:,Attr])
        values=copy(sorted(values))
        #print 'test----',values
        toTryThreshholds=[]
        if(len(values)==1):
            x = float("-inf")
            return x,x
		
        for i in range(0,len(values)-1):
           toTryThreshholds.append((values[i]+values[i+1])/2)
        
        toTryThreshholds=set(toTryThreshholds)
        #print 'toTryThreshholds is ',toTryThreshholds
        if Attr in self.usedThresholds:
            for used in self.usedThresholds[Attr]:
                if used in toTryThreshholds:
                    toTryThreshholds.remove(used)
        
        #now we have all the thresholds that we need to try
        toTryThreshholds=copy(sorted(toTryThreshholds))
        IG=[]
        
        
        for threshold in toTryThreshholds:
            IG.append(self.findIG(data,threshold,Attr,labels))
        
        maxIG=max(IG)
        maxThresh=IG.index(maxIG)
        
        return toTryThreshholds[maxThresh],maxIG    
	
    def findIG(self,data1,threshold,Attr,labels1):
		data=copy(data1)
		labels=copy(labels1)
		#print labels
		rowsLeft=np.where(data[:,Attr]>=threshold)[0]
		rowsRight=np.where(data[:,Attr]<threshold)[0]
        
        #Calculate parent threshold 
		rowsH=np.where(labels== 1 )[0]
		rowsC=np.where(labels== -1 )[0]
		pH=float(rowsH.shape[0])/labels.shape[0]
		pC=float(rowsC.shape[0])/labels.shape[0]
        
		if pH==0 or pC==0:
			HX=0
		else:
			HX=-1*pH*np.log2(pH) - pC*np.log2(pC)
        
        #now calculate the H(Y|X)
        #print 'in IG labels.shape is ',labels.shape
        
		#labelsLeft=[]
		#labelsRight=[]
		labelsLeft=copy(labels[rowsLeft])
		labelsRight=copy(labels[rowsRight])
		#print labelsLeft
		#print labelsRight
        #print 'in IG labelsLeft.shape is ',labelsLeft.shape
        #print 'in IG labelsRight.shape is ',labelsRight.shape
        #For Left Child

		rowsH=np.where(labelsLeft== 1)[0]
		rowsC=np.where(labelsLeft== -1)[0]
        
		pHL=float(rowsH.shape[0])/labelsLeft.shape[0]
		pCL=float(rowsC.shape[0])/labelsLeft.shape[0]
		if pHL==0 or pCL==0:
			HY_X_L=0
		else:
			HY_X_L=-1*pHL*np.log2(pHL) - pCL*np.log2(pCL)
			HY_X_L=HY_X_L*float(rowsLeft.shape[0])/data.shape[0]
        
        
        #For Right Child

		rowsH=np.where(labelsRight==1)[0]
		rowsC=np.where(labelsRight==-1)[0]
        
        #print 'labelsRight.shape[0] is ',labelsRight.shape[0]
		pHR=float(rowsH.shape[0])/labelsRight.shape[0]
		pCR=float(rowsC.shape[0])/labelsRight.shape[0]
        
		if pHR==0 or pCR==0:
			HY_X_R=0
		else:
			HY_X_R=-1*pHR*np.log2(pHR) - pCR*np.log2(pCR)
			HY_X_R=HY_X_R*float(rowsRight.shape[0])/data.shape[0]
        
		IG=HX-HY_X_L-HY_X_R
		#print HX
		return IG 
    
    def getTopic(self):
       return self.topics         
            
                             
       
    #def checkValues(self):

if __name__ == '__main__':       
	
	csvname = './white_entropy_en_pos.csv'
	df = pd.read_csv(csvname)
	ob1=DecisionTree(df,10)
	ob1.loadTrain(df)
	#ob1.loadTest()

