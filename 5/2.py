# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:14:20 2021

@author: 28153
"""
'''
2. 采用局部加权线性回归，预测鲍鱼年龄（使用鲍鱼年龄数据集abalone.txt），
随机取部分数据用于训练，余下数据测试。采用不同核大小（不同k值），
分别计算训练误差和测试误差。
'''
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

# 使用局部加权线性回归计算回归系数w
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))                                   
    for j in range(m):                                              
        diffMat = testPoint - xMat[j, :]                                 
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)                                        
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  #计算回归系数!!!!!
    return testPoint * ws

# 局部加权线性回归测试
def lwlrTest(testArr, xArr, yArr, k):  
    m = np.shape(testArr)[0]                                       
    yHat = np.zeros(m)    
    for i in range(m):                                             
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# 计算回归系数w
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) **2).sum()


if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    print(len(abX))
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1,error=',rssError(abY[100:199], yHat01.T))
    print('k=1  时,error=',rssError(abY[100:199], yHat1.T))
    print('k=10 时,error=',rssError(abY[100:199], yHat10.T))
      
        
    


















