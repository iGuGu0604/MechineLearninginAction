# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:08:35 2021

@author: 28153
"""
'''
3.利用机器学习库sklearn中的随机森林分类器
RandomForestClassifier对Iris数据集进行交叉验证，测试其准确率。
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

for i in range(2,5):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1*i, random_state=0)
    clf = RandomForestClassifier()
    clf = clf.fit(X_train, y_train)
    cvs = cross_val_score(clf,X_test,y_test)
    print(cvs)























