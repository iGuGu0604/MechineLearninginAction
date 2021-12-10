# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:52:55 2021

@author: 28153
"""
'''
1.对于Iris数据集（sklearn库自带鸢尾花数据集），试采用Bagging方法如：
随机森林以及Boosting方法如：Adaboost和SVM分别进行分类
（采用sklearn库或者自编python代码均可），对比几种算法的训练集误差、
测试集误差和运行时间。
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn import datasets
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target, test_size=0.7, random_state=0)
#=======random forest========
since=time.time()
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=10000)
rf.fit(X_train, y_train)
fit_rf = time.time()-since
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
predicted_rf=time.time()-since
print('Out-of-bag score estimate: ',rf.oob_score_)
print('Mean accuracy score:',accuracy)
print("rf_fit-time:",fit_rf)
print("rf_predicted-time",predicted_rf-fit_rf)
print("========================")
#=======adaboost========
since=time.time()
ada = AdaBoostClassifier(n_estimators=100,random_state=10000)
ada.fit(X_train, y_train)
fit_ada = time.time()-since
predicted = ada.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
predicted_ada=time.time()-since
print('Mean accuracy score:',accuracy)
print("ada_fit-time:",fit_ada)
print("ada_predicted-time",predicted_ada-fit_ada)
print("========================")
#=======SVM========
since=time.time()
svm = SVC(kernel="rbf",random_state=1000)
svm.fit(X_train, y_train)
fit_svm = time.time()-since
predicted = svm.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
predicted_svm=time.time()-since
print('Mean accuracy score:',accuracy)
print("svm_fit-time:",fit_svm)
print("svm_predicted-time",predicted_svm-fit_svm)







