import numpy
from pandas import read_csv
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_file():
   data = read_csv('secom.data', sep=' ', names=[i for i in range(590)])
   data = numpy.array(data)
   for i in range(data.shape[1]):
       temp = numpy.array(data)[:, i].tolist()
       mean = numpy.nanmean(temp)
       data[numpy.argwhere(numpy.isnan(data[:, i].T)), i] = mean
   return data

def main():
    data=load_file()
    total=0
    y=list()
    for i in range(30):
        pca = PCA(n_components=i)
        pca.fit(data)
        y.append(sum(pca.explained_variance_ratio_))
        print("前", i + 1, "个特征对应的方差百分比之和为", sum(pca.explained_variance_ratio_), "%")
    x = list(range(30))
    plt.plot(x, y)
    plt.show()

main()

