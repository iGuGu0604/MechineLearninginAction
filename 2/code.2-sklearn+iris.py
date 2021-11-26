# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:25:39 2021

@author: 28153
"""
'''
2 利用机器学习库sklearn中的决策树分类器DecisionTreeClassifier对Iris数据集进行交叉验证，
测试其准确率。
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split

figure(figsize=(30, 30), dpi=80)

iris = load_iris()
X, y = iris.data, iris.target
for i in range(2,5):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1*i, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    cvs = cross_val_score(clf,X_test,y_test)
    print(cvs)
#tree.plot_tree(clf)






