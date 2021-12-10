# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:37:27 2021

@author: 28153
"""
'''
3. 使用二分k-means算法对地图上的点聚类
使用places.txt文件中地图上的点的纬度和经度数据（第4列、第5列），
采用二分k-means对地图上的点聚类（k设定为5）。
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataSet=[]
    place=[]
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        #fltLine = list(map(float,curLine)) #使用map函数将curLine里的数全部转换为float型
        place.append(float(curLine[3]))
        place.append(float(curLine[4]))
        dataSet.append(place) 
        place=[]
    return dataSet
  
#print(loadDataSet("places.txt"))  
# 计算两个向量的欧式距离
def culDistance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) 

# 为给定数据集构建一个包含k个随机质心的集合,是以每列的形式生成的
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) 
    for j in range(n):  
        minJ = min(dataSet[:,j])  # 找到每一维的最小
        rangeJ = float(max(dataSet[:,j]) - minJ) # 每一维的最大和最小值之差
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) # 生成随机值
        #print centroids[:,j]
    return centroids  # 返回随机质心,是和数据点相同的结构

# k--均值聚类算法(计算质心--分配--重新计算)
def kMeans(dataSet, k, distMeas=culDistance, createCent=randCent): # k是簇的数目
    m = shape(dataSet)[0]  # 得到样本的数目
    clusterAssment = mat(zeros((m,2))) #  创建矩阵来存储每个点的簇分配结果
                                       #  第一列：记录簇索引值，第二列：存储误差，欧式距离的平方
    centroids = createCent(dataSet, k)  # 创建k个随机质心
    clusterChanged = True
    while clusterChanged:  # 迭代使用while循环来实现
        clusterChanged = False  
        for i in range(m):  # 遍历每个数据点，找到距离每个点最近的质心
            minDist = inf; minIndex = -1
            for j in range(k):  # 寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: # 更新停止的条件
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 # minDist**2就去掉了根号         
        for cent in range(k):  # 更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] 
            centroids[cent,:] = mean(ptsInClust, axis=0) # 然后计算均值，axis=0:沿列方向 
    #print 'centroids:',centroids
    return centroids, clusterAssment # 返回簇和每个簇的误差值，误差值是当前点到该簇的质心的距离

def biKmeans(dataSet, k, distMeas=culDistance):
    m = shape(dataSet)[0] 
    clusterAssment = mat(zeros((m,2))) #创建一个矩阵来储存数据中每个点的簇分配结果及平方误差  
    centroid0 = mean(dataSet, axis=0).tolist()[0]  
    centList =[centroid0]  
    for j in range(m):   
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): 
        lowestSSE = inf   
        for i in range(len(centList)): 
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # centroidMat是矩阵
            sseSplit = sum(splitClustAss[:,1])                                                  
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])# 所有剩余数据集的误差之和           
            if (sseSplit + sseNotSplit) < lowestSSE: # 划分后的误差和小于当前的误差，本次划分被保存                
                bestCentToSplit = i  
                bestNewCents = centroidMat                 
                bestClustAss = splitClustAss.copy()  
                lowestSSE = sseSplit + sseNotSplit   
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit        
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]  
        centList.append(bestNewCents[1,:].tolist()[0]) # centList是列表的格式
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment # 返回质心列表和簇分配结果
        
dataSet = mat(loadDataSet('places.txt'))
myCentList,myNewAssment = biKmeans(dataSet,5)
print ("最终质心：\n",myCentList)
print ("索引值和均值：\n",myNewAssment) 
dataSet=dataSet.tolist()
myNewAssment=myNewAssment.tolist()
print(myNewAssment)

red_x=[];red_y=[]
orange_x=[];orange_y=[]
yellow_x=[];yellow_y=[] 
green_x=[];green_y=[]
blue_x=[];blue_y=[]
for i in range(len(myNewAssment)):
    #print(myNewAssment[i][0])
    if myNewAssment[i][0]==0.0:
        red_x.append(dataSet[i][0])
        red_y.append(dataSet[i][1])
    elif myNewAssment[i][0]==1.0:
        orange_x.append(dataSet[i][0])
        orange_y.append(dataSet[i][1])
    elif myNewAssment[i][0]==2.0:
        yellow_x.append(dataSet[i][0])
        yellow_y.append(dataSet[i][1])
    elif myNewAssment[i][0]==3.0:
        green_x.append(dataSet[i][0])
        green_y.append(dataSet[i][1]) 
    else:
        blue_x.append(dataSet[i][0])
        blue_y.append(dataSet[i][1])
    
#print(green_x)
plt.scatter(red_x,red_y,color="red")
plt.scatter(orange_x,orange_y,color="orange")
plt.scatter(yellow_x,yellow_y,color="yellow")
plt.scatter(green_x,green_y,color="green")
plt.scatter(blue_x,blue_y,color="blue")
plt.show()


    