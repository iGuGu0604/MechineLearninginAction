# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:48:48 2021

@author: 28153
"""
'''
1. 利用 PCA 对半导体制造数据集 secom.data
 (http://archive.ics.uci.edu/ml/machine-learning-databases/secom)进行降维。
 注:该数据包含了较多缺失值，采用平均值对所有缺失值进行替换，平均值由非缺失的数据得到。
 在实验中取不同的主成分截断值来检验性能。采用自编 python 代码和使用 sklearn库分别实现。
'''
from numpy import *
import pandas as pd
from

def loadDataSet():
    data = pd.read_csv('secom.data', sep=' ', names=[i for i in range(590)])
    data = np.array(data)

    for i in range(data.shape[1]):
        temp = array(data)[:, i].tolist()
        mean = nanmean(temp)
        data[argwhere(isnan(data[:, i].T)), i] = mean
    return data

def pca(dataMat,topNfeat=9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat*redEigVects)
    reconMat = (lowDDataMat*redEigVects.T)+meanVals
    return lowDDataMat,reconMat    
      
 
dataMat,reconMat = pca(loadDataSet())
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
print(meanRemoved)
covMat = cov(meanRemoved,rowvar=0) 
eigVals,eigVects = linalg.eig(mat(covMat))
print(eigVals)   
