# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:20:44 2021

@author: 28153
"""

#导入模块，调用相关函数
from numpy import *
import operator
from os import listdir
from matplotlib import pyplot as plt

#准备数据：从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    print(arrayOfLines)
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    #fr = 
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3] #三个特征存放到矩阵中
        classLabelVector.append((listFromLine[-1])) #类别标签 ##为什么是[-1]
        index += 1
    return returnMat,classLabelVector
#print(file2matrix(r"C:\Users\28153\Desktop\exercise1\datingTestSet2.txt"))

#准备数据，对训练数据创建数据集和标签
dataSet,labels = file2matrix(r"C:\Users\28153\Desktop\MachineLearning\exercise1\datingTestSet2.txt")
inX = zeros((1,dataSet.shape[1]))
k = 10
inX = [56789,10.833,7.62394]
print(inX)
'''
print(dataSet)
print("1/2")
print(labels)
'''
'''
函数功能：knn分类
input：inX（测试集：用来检验最终成果）（1xN）
dataSet：已知数据的特征（NxN）
labels:已知数据的标签或类别（1xM vector）
k:k近邻算法中的k
Output：测试样本最可能所属的标签
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet =normDataSet/(tile(ranges,(m,1)))
    return normDataSet,ranges,minVals

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

print(classify0(inX,dataSet,labels,k))

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix(r"C:\Users\28153\Desktop\MachineLearning\exercise1\datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:",classifierResult,"  the real answer is: ",datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount+=1.0
    print("the total error rate is ",errorCount/float(numTestVecs))
  
datingClassTest()    
  
#Analyze data
datingDataMat = dataSet
datingLabels = labels
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('video games')
plt.ylabel('ice cream')
print(len(datingLabels))
for i in range(len(datingLabels)):
    if(datingLabels[i]=='1'):
        type1 = ax.scatter(datingDataMat[i,1],datingDataMat[i,2],marker = "x",s=10,color="blue")
    if(datingLabels[i]=='2'):
        type2 = ax.scatter(datingDataMat[i,1],datingDataMat[i,2],marker = "o",s=20,color="green")    
    if(datingLabels[i]=='3'):
        type3 = ax.scatter(datingDataMat[i,1],datingDataMat[i,2],marker = "^",s=30,color="red")        
plt.title("dating statistics")
plt.legend((type1,type2,type3),("1","2","3"))
plt.show()

def calssifyPerson():
    resultList = ["not at all","in small does","in large does"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix(r"C:\Users\28153\Desktop\MachineLearning\exercise1\datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[int(classifierResult)-1])
calssifyPerson()

    
    
    
    
    
    
    
    
    
    
    
    