# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:46:46 2021

@author: 28153
"""
'''
1. 利用 PCA 对半导体制造数据集 secom.data
 (http://archive.ics.uci.edu/ml/machine-learning-databases/secom)进行降维。
 注:该数据包含了较多缺失值，采用平均值对所有缺失值进行替换，平均值由非缺失的数据得到。
 在实验中取不同的主成分截断值来检验性能。采用自编 python 代码和使用 sklearn库分别实现。
'''
import pandas as pd
import numpy as np
from numpy import linalg

def load_file():
   data = pd.read_csv('secom.data', sep=' ', names=[i for i in range(590)])
   data = np.array(data)

   for i in range(data.shape[1]):
       temp = np.array(data)[:, i].tolist()
       mean = np.nanmean(temp)
       data[np.argwhere(np.isnan(data[:, i].T)), i] = mean

   return data


def pca(K):
    X = load_file()
    N = X.shape[0]
    En = np.eye(N)
    In = np.ones((N, 1), float)

    H = En - (1/N)*np.dot(In, In.T)  #定义中心矩阵
    S = (1/N)*np.dot(np.dot(X.T, H), X)  #定义协方差矩阵

    val, vec = linalg.eig(S)   #求解特征值与特征向量
    sorted_indices = np.argsort(-val)   #从大到小排序
    #取前K个最大的特征值的特征向量
    final = np.zeros((K, vec.shape[1]), float)
    for i in range(K):
        final[i, :] = vec[sorted_indices[i], :]

    final_data = np.dot(X, final.T)  #降为K为后的矩阵
    return final_data


if __name__ == '__main__':
    K = 250
    print(pca(K))
