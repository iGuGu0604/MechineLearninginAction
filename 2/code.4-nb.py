# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 17:08:12 2021

@author: 28153
"""
'''
4. 使用email文件夹下spam文件夹(垃圾邮件)和ham文件夹（正常邮件）的共50个邮件，
采用交叉验证的方式，随机选取10个文件作为测试数据，其他作为训练数据，
采用朴素贝叶斯进行垃圾邮件的分类。计算多次迭代后的平均错误率。
'''
import numpy as np
import random
import re


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
"""
function classification:
    train the method
parameters:
    trainMatrix,traiCategory
return:
    p0Vect,p1Vect,pAbusive
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom +=sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
'''
function classification:
    classify the email
parameters:
    vec2Classify,p0Vec,p1Vec,pClass1
return 0/1
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
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

'''
funcion classification:
    test
parameter:none
return:none
'''
def spamTest():
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
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))#训练朴素贝叶斯模型
    errorCount = 0                                                        #错误分类计数
    for docIndex in testSet:                                              #遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])         #测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:#如果分类错误
            errorCount += 1                                               #错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    spamTest()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     