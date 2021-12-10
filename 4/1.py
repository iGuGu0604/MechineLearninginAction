# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:53:23 2021

@author: 28153
"""

'''
1.使用支持向量机完成sklearn自带的digits数据集进行手写数字的识别。尝试采用不同核函数、
调整不同gamma，C参数值等，采用交叉验证法，分别计算训练集错误率和测试集错误率，并进行调参分析。
注:可使用sklearn库的SVM实现.
'''

# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC

"""
Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量
"""

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
'''
c 惩罚参数
kernel 核函数  'linear’：线性核函数\\‘poly’：多项式核函数\\‘rbf’：径像核函数/高斯核\\
    ‘sigmod’：sigmod核函数\\‘precomputed’：核矩阵
gamma 核函数系数 只对’rbf’ ,’poly’ ,’sigmod’有效。
'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits/')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % (fileNameStr))
    clf = SVC(C=200,kernel='poly',gamma='scale')  
    clf.fit(trainingMat,hwLabels)
    testFileList = listdir('digits/testDigits/')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % (fileNameStr))
        classifierResult = clf.predict(vectorUnderTest)
        #print("分类结果%d\t真实结果%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("错误个数：%d\n错误率：%f%%" % (errorCount, errorCount/mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
