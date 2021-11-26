# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:10:22 2021

@author: 28153
"""
'''
3.通过访问：
http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data 
中提供的病马原始数据，采用sklearn.impute 中SimpleImputer对原始缺失数据进行处理（处理策略不限定，
如：特殊值等）。
'''
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

data = pd.read_fwf("horse-colic.data")
#data2 = pd.read_cvs("horse-colic.data")
print(data)

data_replace = data.replace('?',np.NaN,inplace=False)
print(data_replace)

data1 = np.load("horse-colic.data",allow_pickle=True)
print(data1)
'''
data_new = data_replace.to_numpy()
print(data_new)
'''
'''
fill_Nan = SimpleImputer(missing_values="?",strategy="mean")
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
'''