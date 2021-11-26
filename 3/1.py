# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:19:32 2021

@author: 28153
"""
'''
1. 使用horseColicTraining.txt文件作为训练集（每行包含了病马的20个特征和是否死亡的标签），
horseColicTest.txt作为测试集，利用Logistic回归预测病马的死亡率。
计算多次迭代后的平均错误率。
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))



def stocGradAscent1(dataMatrix, classLabels, numIter=500):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化	#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 															#返回

def colicTest():
	frTrain = open('C:\\Users\\28153\\Desktop\\MachineLearning\\exercise 3\\horseColicTraining.txt')										#打开训练集
	frTest = open('C:\\Users\\28153\\Desktop\\MachineLearning\\exercise 3\\horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)		#使用改进的随即上升梯度训练
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec) * 100 								#错误率计算
	print("测试集错误率为: %.2f%%" % errorRate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
colicTest()