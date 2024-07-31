# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:18:40 2024

@author: alexa
"""

from sklearn.preprocessing import normalize as nm
class cleaning:
    def __init__(self):
        
        pass
    
    def setAllTypeFloat(dataset):
        for i in dataset:
            try:
                dataset[i]=dataset[i].astype(float)
                pass
            except:
                pass
            pass
        return dataset
    
    def normalization(dataColumn, normType):
        normData=nm([dataColumn],norm=normType)
        if normType=='l1':
            normRate=sum(dataColumn**2)**0.5
            pass
        elif normType=='l2':
            normRate=sum(dataColumn)
            pass
        else:
            normRate=max(dataColumn)
            pass
        return [normData, normRate]
    
    def kvantil(dataset, fields, kvantUp, kvantDown):
        upPercent=float(kvantUp.replace('%', ''))/100
        downPercent=float(kvantDown.replace('%', ''))/100
        for i in fields:
            upValue=dict(dataset[i].describe([upPercent]))[kvantUp]
            downValue=dict(dataset[i].describe([downPercent]))[kvantDown]
            dataset=dataset[dataset[i]<=upValue][dataset[i]>=downValue]
            pass
        return dataset
        pass
    
  
    def strToIndex(columns):
        replace_list=[]
        for column in columns:
            st=list(set(column))
            for i in st:
                column.replace(i, st.index(i), inplace=True)
                replace_list.append([st.index(i), i])
                pass
            pass
        return replace_list
        pass
    
    
    
    pass