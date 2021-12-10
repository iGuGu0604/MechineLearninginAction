# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 15:55:56 2021

@author: 28153
"""
'''
1.使用支持向量机完成sklearn自带的digits数据集进行手写数字的识别。尝试采用不同核函数、
调整不同gamma，C参数值等，采用交叉验证法，分别计算训练集错误率和测试集错误率，并进行调参分析。
注:可使用sklearn库的SVM实现.
'''
'''
c 惩罚参数
kernel 核函数  'linear’：线性核函数\\‘poly’：多项式核函数\\‘rbf’：径像核函数/高斯核\\
    ‘sigmod’：sigmod核函数\\‘precomputed’：核矩阵
gamma 核函数系数 只对’rbf’ ,’poly’ ,’sigmod’有效。
'''
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

def loadDigits():
    X,y = load_digits(return_X_y = True)
    #print(len(X),y)
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #print(len(X_train), len(X_test), len(y_train), len(y_test))
    return X_train,X_test,y_train,y_test

def testFuc():
    X_train,X_test,y_train,y_test = loadDigits()
    kernel=['linear','poly','rbf','sigmoid']
    C=[1.0,10.0,100.0]
    for k in kernel:
        print("\n======kernel:%s======"%k)
        if k =='linear' or k=='precomputed':
            for c in C:
                print("-----C=%.1f-----"%c)
                svm=SVC(C=c,kernel=k)
                begin=time.time()
                svm.fit(X_train,y_train)
                score=svm.score(X_test,y_test)
                end=time.time()-begin
                print("score=",score)
                print("time=%.5f\n"%end)
        else:
            for c in C:
                print("-----C=%.1f-----"%c)
                print("---gamma=‘auto’---")
                svm=SVC(C=c,kernel=k,gamma='auto')
                begin=time.time()
                svm.fit(X_train,y_train)
                score=svm.score(X_test,y_test)
                end=time.time()-begin
                print("score=",score)
                print("time=%.5f\n"%end)
                print("---gamma=‘scale’---")
                svm=SVC(C=c,kernel=k,gamma='scale')
                begin=time.time()
                svm.fit(X_train,y_train)
                score=svm.score(X_test,y_test)
                end=time.time()-begin
                print("score=",score)
                print("time=%.5f\n"%end)
 
testFuc()
            
    
            
    
            
    
            
    
            
    
            
    
            
    
            