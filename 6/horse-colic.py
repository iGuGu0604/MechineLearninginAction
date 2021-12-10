# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:56:23 2021

@author: 28153
"""
'''
3.（选做题**）对于病马死亡率预测问题，针对该数据集、尝试
目前为止学习到的统计机器学习方法（SVM, Bagging,Boosting 等
等）或采用神经网络（BP 神经网络等），或尝试不同的缺失值填
补方法（插值、自行设计机器学习算法等等），看看是否能在本
书前述实验基础上进一步提升预测准确率。
说明:准确率在前 5 名的同学将计入平时成绩加权。本题在考试
前均有效。
'''
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from numpy import linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
import random

data = pd.read_fwf("horse-colic.data",header=None)
data.columns=["1","2","3"]
data2 = data['3'].str.split(" ",expand=True)
data=data.drop(["3"],axis=1)
c = []
for i in range(3,29):
    c.append("%d"%i)
data2.columns =c
data0 = pd.concat([data,data2],axis=1)
data0 = data0.replace("?",np.nan)
#print(data0)

#------mean
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
data_transform = imp_mean.fit_transform(data0)
#已经将原始数据用均值替代
#print(type(data_transform))
#np.savetxt('3.mean.txt', data_transform,delimiter='    ', fmt='%.2f')
Labelclass=[]
for i in range(len(data_transform)):
    Labelclass.append(data_transform[i][-1])
print(type(data_transform))
feature=np.delete(data_transform,[-1],1)
feature=np.delete(feature,[2],1)
#print(np.shape(feature),np.shape(data_transform))
#尝试对数据用PCA降维再分类
'''
for i in range(27):
    pca = PCA(n_components=i)
    pca.fit(feature)
    print("前", i + 1, "个特征对应的方差百分比之和为", sum(pca.explained_variance_ratio_),"%")

meanVals = np.mean(feature, axis=0)
meanRemoved = feature - meanVals
covMat = np.cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(np.mat(covMat))
print(eigVals)
'''
#测试后将病马集降维到
pca=PCA(n_components=22)
newFeature=pca.fit_transform(feature)
#np.savetxt('afterpca.txt', newFeature,delimiter='    ', fmt='%.2f')
'''
#knn 正确率0.63
neigh = KNeighborsClassifier(n_neighbors=4)
X_train, X_test, y_train, y_test = train_test_split(newFeature,Labelclass,test_size=0.33, random_state=42)
neigh.fit(X_train, y_train)
score=neigh.score(X_test, y_test)
print(score)
'''
est = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform')
feature_discretizer = est.fit_transform(feature)
'''
X_train, X_test, y_train, y_test = train_test_split(newFeature,Labelclass,test_size=0.33, random_state=42)
clf = RandomForestClassifier(max_depth=20, random_state=999)
clf.fit(X_train, y_train)
score=clf.score(X_test, y_test)
print("离散化前",score)
'''
'''
sum=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(feature_discretizer,Labelclass,test_size=0.33, random_state=773)
    clf = RandomForestClassifier(max_depth=20, random_state=19999)
    clf.fit(X_train, y_train)
    score=clf.score(X_test, y_test)
    sum+=score
    print(score)
print("平均正确率",sum/10)
'''
sum=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(feature,Labelclass,test_size=0.33, random_state=random.randint(700,1000))
    #print(X_train[30])
    clf = RandomForestClassifier(max_depth=20, random_state=19999)
    clf.fit(X_train, y_train)
    score=clf.score(X_test, y_test)
    sum+=score
    print(score)
print("平均正确率",sum/10)
'''
#------median
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
data_transform = imp_median.fit_transform(data0)
print(data_transform)
np.savetxt('3.median.txt', data_transform,delimiter='    ', fmt='%.1f')
#------most frequentmost_frequent
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_transform = imp_most_frequent.fit_transform(data0)
print(data_transform)
np.savetxt('3.most_frequent.txt', data_transform,delimiter='    ', fmt='%14s')
'''
