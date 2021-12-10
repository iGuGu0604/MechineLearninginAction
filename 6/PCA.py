'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
import numpy
from pandas import read_csv
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return numpy.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = numpy.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = numpy.cov(meanRemoved, rowvar=0)
    eigVals,eigVects = numpy.linalg.eig(numpy.mat(covMat))
    eigValInd = numpy.argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = numpy.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = numpy.mean(datMat[numpy.nonzero(~numpy.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[numpy.nonzero(numpy.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

def replaceNanWithMean2():
    #datMat = loadDataSet('secom.data', ' ')
    url = 'secom.data'
    dataframe = read_csv(url, sep='\s+', header=None, na_values=',',na_filter=True)
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1]
    print('Missing: %d' % sum(numpy.isnan(X).flatten()))
    print(data.shape)
    print(data)
    imp=SimpleImputer(strategy='mean')
    print("############################################################")
    print(data)
    imp.fit(data)

    print(data)
    return data

def load_file():
   data = read_csv('secom.data', sep=' ', names=[i for i in range(590)])
   data = numpy.array(data)
   for i in range(data.shape[1]):
       temp = numpy.array(data)[:, i].tolist()
       mean = numpy.nanmean(temp)
       data[numpy.argwhere(numpy.isnan(data[:, i].T)), i] = mean
   return data

def main():
    dataMat=load_file()
    print(dataMat.shape)
    meanVals=numpy.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=numpy.cov(meanRemoved,rowvar=0)
    eigVals,eigVects=numpy.linalg.eig(numpy.mat(covMat))
    #print(eigVals)
    total=0
    y = list()
    for i in range(30):
        total+=eigVals[i]
        print("前",i+1,"个特征对应的方差百分比之和为",(total/sum(eigVals)*100).real,"%")
        y.append((total/sum(eigVals)*100).real)
    x=list(range(30))
    # for i in range(30):
    #     y.append((eigVals[i]/sum(eigVals)*100).real)
    plt.plot(x, y)
    plt.show()
    #print(eigVals[0]/sum(eigVals)*100,"%")






main()