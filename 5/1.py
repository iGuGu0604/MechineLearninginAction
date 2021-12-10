# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:14:19 2021

@author: 28153
"""
'''
1.对于Iris数据集（sklearn库自带鸢尾花数据集），试采用Bagging方法如：随机森林
以及Boosting方法如：Adaboost和SVM分别进行分类（采用sklearn库或者自编python代码均可），
对比几种算法的训练集误差、测试集误差和运行时间。
'''

import  numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.ensemble import AdaBoostClassifier #Adaboost
from sklearn.svm import SVC
from sklearn.datasets import load_iris #鸾尾花数据
import time
 
RF = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)
Ad = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, algorithm='SAMME.R')
svm=SVC()
iris = load_iris()
x = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print("========")
#随机森林
starttime = time.time()
RF.fit(X_train,y_train)
print('RandomForestClassifier:', RF.score(X_test, y_test))
print('RandomForestClassifier:', RF.score(X_train,y_train))
endtime = time.time()
print(endtime - starttime)
print("========")
#Adaboost
starttime = time.time()
Ad.fit(X_test, y_test)
print('AdaboostClassifier:', Ad.score(X_train,y_train))
print('AdaboostClassifier:', Ad.score(X_test, y_test))
endtime = time.time()
print(endtime - starttime)
print("========")
#支持向量机
starttime = time.time()
svm.fit(X_train,y_train)
print('SVM:', svm.score(X_test, y_test))
print('SVM:', svm.score(X_train,y_train))
endtime = time.time()
print(endtime - starttime)




