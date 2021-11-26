# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:13:57 2021

@author: 28153
"""
'''
5.采用sklearn中的MultinomialNB朴素贝叶斯算法对sklearn的datasets中
自带的digits数据集进行分类，测试分类效果。
'''

from sklearn import datasets, model_selection, naive_bayes
import matplotlib.pyplot as plt
import numpy as np


def load_data(datasets_name='iris'):
    if datasets_name == 'iris':
        data = datasets.load_iris()  # 加载 scikit-learn 自带的 iris 鸢尾花数据集-分类
    elif datasets_name == 'wine': # 0.18.2 没有
        data = datasets.load_wine()  # 加载 scikit-learn 自带的 wine 红酒起源数据集-分类
    elif datasets_name == 'cancer':
        data = datasets.load_breast_cancer()  # 加载 scikit-learn 自带的 乳腺癌数据集-分类
    elif datasets_name == 'digits':
        data = datasets.load_digits()  # 加载 scikit-learn 自带的 digits 糖尿病数据集-回归
    elif datasets_name == 'boston':
        data = datasets.load_boston()  # 加载 scikit-learn 自带的 boston 波士顿房价数据集-回归
    else:
        pass
    return model_selection.train_test_split(data.data, data.target,test_size=0.25, random_state=0,stratify=data.target) 
    # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4
    
def test_MultinomialNB(*data, show=False):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train, y_train)
    # print('MultinomialNB Training Score: %.2f' % cls.score(X_train, y_train))
    print('MultinomialNB Testing Score: %.2f' % cls.score(X_test, y_test))

def testFuc(fuc,show = False,no_all = True):
    for i in ['iris', 'wine', 'cancer', 'digits', ]:
            print('\n====== %s ======\n' % i)          
            X_train, X_test, y_train, y_test = load_data(datasets_name=i)  # 产生用于分类问题的数据集
            if no_all:
                fuc(X_train, X_test, y_train, y_test,show = show) 
            else:
                test_GaussianNB(X_train, X_test, y_train, y_test,show = show)  # 调用 test_GaussianNB
                test_MultinomialNB(X_train,X_test,y_train,y_test,show = show) # 调用 test_MultinomialNB
                test_MultinomialNB_alpha(X_train, X_test, y_train, y_test,show = show)  # 调用 test_MultinomialNB_alpha
                test_BernoulliNB(X_train,X_test,y_train,y_test,show = show) # 调用 test_BernoulliNB
                test_BernoulliNB_alpha(X_train, X_test, y_train, y_test,show = show)  # 调用 test_BernoulliNB_alpha
                test_BernoulliNB_binarize(X_train, X_test, y_train, y_test,show = show)  # 调用 test_BernoulliNB_binarize
                
testFuc(test_MultinomialNB)