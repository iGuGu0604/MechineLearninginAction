# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:57:28 2021

@author: 28153
"""
'''
2.采用sklearn.linear_model.LogisticRegression实现上述数据集Logistic回归预测病马死亡率。
'''

from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def colicSklearn():
	frTrain = open('C:\\Users\\28153\\Desktop\\MachineLearning\\exercise 3\\horseColicTraining.txt')										#打开训练集
	frTest = open('C:\\Users\\28153\\Desktop\\MachineLearning\\exercise 3\\horseColicTest.txt')												#打开测试集	
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('正确率:%f%%' % test_accurcy)
colicSklearn()

