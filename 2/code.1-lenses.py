# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:44:36 2021

@author: 28153
"""
'''
预测隐形眼镜类型使用lenses.txt中的隐形眼镜数据集，
采用第三章中介绍的ID3算法构建决策树。
使用决策树，输入几组隐形眼镜特征数据，
例如：'young','hyper','no','reduced'，'pre','hyper','no','normal'；等进行测试，
预测隐形眼镜类型。
'''
import numpy as np
from math import log
import operator

"""
函数说明：加载隐形眼镜的数据集
function clarfication: load the dataSet of contact lenses

parameters:
    filename
return:
    dataSet(list)
"""
def loaddataSet(filename):
    fr = open(filename)
    dataSet = []
    for line in fr.readlines():
        dataSet.append(line.strip().split("\t"))
    return dataSet
#print(len(loaddataSet("lenses.txt")))
#print(loaddataSet("lenses.txt"))

"""
函数说明：计算给定数据集的香农熵
Parameters:
    dataSet
Returns:
    shannonEnt 
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)   #数据集的行数：len()的功能:返回数据的长度
    labelCounts = {}            #保存每个标签的出现次数
    for featVec in dataSet:
        #print(featVec[-1])
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntires
        shannonEnt -= prob*log(prob,2)
    #print(labelCounts)
    return shannonEnt
#print(calcShannonEnt(loaddataSet("lenses.txt")))

"""
函数说明：划分数据集
parameter:
    dataSet,axis,value
return:
    retDataSet
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  #切片操作
            reducedFeatVec.extend(featVec[axis+1:])  #extend用于在列表末尾一次性追加另一个序列中的多个值
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
function classification:
    choose the best feature to split
parameters:
    dataSet
returns:
    bestfeature
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1          #特征数量
    baseEntropy = calcShannonEnt(dataSet)   #计算数据集的香农熵
    bestInfoGain = 0.0                      #信息增益
    bestFeature = -1                        #最优特征的索引值
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]    
        uniqueVals = set(featList)          #创建set集合{}，元素不可重复
        newEntropy = 0.0                    #经验条件熵
        for value in uniqueVals:            #计算信息增益
            subDataSet = splitDataSet(dataSet,i,value)   #subDataSet划分后的子集
            prob = len(subDataSet)/float(len(dataSet))   
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #print("第%d个特征的增益为%.3f"%(i,infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    #print(bestFeature)
    return bestFeature       
#print(chooseBestFeatureToSplit(loaddataSet("lenses.txt")))

#创建树
'''
函数说明：多数表决函数
当遍历完所有的特征后，类标签仍然不唯一（分支下仍有不同分类的实例）
采用多数表决的方法完成分类
parameters:
    classList
return:
    sortedClassCount[][]
'''
def majorityCnt(classList):
    print("用到了！")
    #创建一个类标签的字典
    classCount = {}
    #遍历类标签列表中的每一个元素
    for vote in classList:
        #如果元素不在字典中
        if vote not in classCount.keys():
            #在字典中添加新的键值对
            classCount[vote]=0
        #否则，当前键对于的值加1
        classCount[vote] += 1
    #对字典中的键对应的值所在的列，按照由大到小进行排序
    #@classCount.items 列表对象
    #@key = operator.intemgetter(1) 获取列表对象的第一个域的值
    #@reverse = true 降序排序，默认是升序排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)    
    #返回出现次数最多的标签
    return sortedClassCount[0][0]

'''
#创建树
parameter:
    dataSet,labels,featLabels
return:
    myTree
'''
def creatTree(dataSet,labels,featLabels):
    #获取数据集中的最后一列的类标签，存入classList列表
    classList = [example[-1] for example in dataSet]
    #通过count()函数获取类标签列表中第一个类标签的数目
    #判断数目是否等于列表长度，相同表面所有类标签相同，属于同一类
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        #多数表决原则，确定类标签
        return majorityCnt(classList)
    #确定出当前最优的分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最优特征
    #print(bestFeat)
    #print(len(labels))
    bestFeatLabel = labels[bestFeat]             #选最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                  #根据最优特征标签生成树
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    #print(featValues)
    uniqueVals = set(featValues)
    #print(uniqueVals)
    for value in uniqueVals:
        subLabels = labels[:]
        #print("labels:",len(labels))
        #print("subLabels:",len(subLabels))
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels,featLabels)                           
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print(featLabels)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat,dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:classLabel = valueOfFeat
    return classLabel

if __name__ == '__main__':
    dataSet = loaddataSet("lenses.txt")
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    featLabels = []
    myTree = creatTree(dataSet, labels,featLabels)
    print(classify(myTree, ['age', 'prescript', 'astigmatic', 'tearRate'], ['young','hyper','no','reduced']))
        
        
        
        
        
        
        
        
        
        
        
        
        