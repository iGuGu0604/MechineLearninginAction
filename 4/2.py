# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:02:34 2021

@author: 28153
"""
'''
在实验2中我们采用了朴素贝叶斯进行了垃圾邮件分类：使用email文件夹下spam文件夹
(垃圾邮件)和ham文件夹（正常邮件）的共50个邮件，采用交叉验证的方式，
随机选取10个文件作为测试数据，其他作为训练数据，采用朴素贝叶斯进行垃圾邮件的分类。
计算多次迭代后的平均错误率。

这里试采用SVM进行垃圾邮件分类，并对两种方法的性能
（训练集错误率、测试集错误率、运行时间等）进行比较。
注:可以使用sklearn库或python实现
'''
import numpy as np
import random
import re
import time
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
'''
function classification:
    creat a list of all the unique words in all of our document
parameters:
    dataSet
return:
    list(vocabSet)
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)
'''
function classification:
     takes the vocabulary list and a document and outputs a vector of 1s and 0s 
     to represent whether a word from our vocabulary is present or not in the given 
     document
parameters:
    vocabList, inputSet
return:
    returnVec
'''
def setOfWords2Vec(vocabSet,inputSet):
    returnVec = [0]*len(vocabSet)
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)]=1
        else:
            print('the word:%s is not in my vocabulary!'%'word')
    return returnVec
'''
function classification:
    test 
parameters:
    none
return:
    none
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W',bigString)
    return[tok.lower() for tok in listOfTokens if len(tok)>2]

def loadDataofEmail():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):                                             #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())#读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                                            #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read()) #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)                    		#标记非垃圾邮件，1表示垃圾文件   
    vocabList = createVocabList(docList)       		#创建词汇表，不重复
    trainingSet = list(range(50)); testSet = []		#创建存储训练集的索引值的列表和测试集的索引值的列表                       
    for i in range(10):#从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))              #随机选取索索引值
        testSet.append(trainingSet[randIndex])                            #添加测试集的索引值
        del(trainingSet[randIndex])                                       #在训练集列表中删除添加到测试集的索引值
    trainMat = []; trainClasses = []                                      #创建训练集矩阵和训练集类别标签系向量             
    for docIndex in trainingSet:                                          #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))     #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                          #将类别添加到训练集类别标签系向量中
    testMat = []; testClasses = []
    for docIndex in testSet:                                              #遍历测试集
        testMat.append(setOfWords2Vec(vocabList, docList[docIndex]))         #测试集的词集模型
        testClasses.append(classList[docIndex])
    return trainMat,testMat,trainClasses,testClasses

def testFunction():
    X_train,X_test,y_train,y_test = loadDataofEmail()
    nb=MultinomialNB()
    svm=SVC()
    print("=====NB=====")
    begin = time.time()
    nb.fit(X_train,y_train)
    score = nb.score(X_test,y_test)
    end=time.time()-begin
    print("socre = %.4f"%score)
    print("time = %.7f"%end)
    print("=====SVM=====")
    begin = time.time()
    svm.fit(X_train,y_train)
    score = svm.score(X_test,y_test)
    end=time.time()-begin
    print("socre = %.4f"%score)
    print("time = %.7f"%end)
    
testFunction()  
  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    