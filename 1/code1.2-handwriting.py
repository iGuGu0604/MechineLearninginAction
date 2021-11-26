# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:56:43 2021

@author: 28153
"""
from numpy import *


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
        return returnVect



def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #shape[0]返回dataSet的行数，应该是1000
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    #print(diffMat.shape[0],diffMat.shape[1])
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)#sum（）求和函数，sum（0）每列所有元素相加，sum（1）每行所有元素相加
    distances = sqDistances**0.5 #开平方，求欧氏距离
    sortedDistIndicies = distances.argsort() 
    #argsort函数返回的是数组值从小到大的索引值
    #print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        #print(classCount[voteIlabel])
    print(classCount)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(r"C:/Users/28153/Desktop/exercise1/digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumberStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(r"C:/Users/28153/Desktop/exercise1/digits/trainingDigits/%s" %fileNameStr)
    testFileList = listdir(r"C:/Users/28153/Desktop/exercise1/digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumberStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r"C:/Users/28153/Desktop/exercise1/digits/trainingDigits/%s" %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:",classifierResult)
        print("the real answer is:",classNumStr)
        if(classifierResult != classNumberStr):
            errorCount+=1.0
    print("\nthe total number of error is: ",errorCount
   # print("\nthe otal error rate is: ",errorCount/float(mTest))       
   '''