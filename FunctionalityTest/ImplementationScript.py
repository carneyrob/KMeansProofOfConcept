import scipy.io as sp
import numpy as np


data = sp.loadmat('kmeans_test_data.mat')['X']


initial_centroids = np.asarray([[3,3],[6,2],[8,5]])

def runKMeans(X, centroids, iters):
    (m,n) = X.shape
    (K,l) = centroids.shape
    idx = np.zeros((m,1))
    for i in range(0,iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,K)
    return (idx,centroids);

def findClosestCentroids(X, centroids):
    (k,d) = centroids.shape
    (m,n) = X.shape
    idx = np.zeros((m,1))
    for i in range(0,m):
        min = 1000000000000
        xr = X[i,:]
        best = 1
        for j in range(0,k):
            p = centroids[j,:]
            sub = np.subtract(p,xr)
            result = np.dot(sub,sub)
            if (result < min):
                best = j
                min = result
        idx[i,0] = best
    return idx

def computeCentroids(X, idx,K):
    (m,n) = X.shape
    centroids = np.zeros((K,n))
    for i in range(0,K):
        countInt = (idx == i).sum().astype(int)
        count = np.zeros((1,n))
        res = np.zeros((1, n))
        for p in range(0,n):
            count[0,p] = countInt
        for j in range(0,m):
            if (idx[j,0]==i):
                res = np.add(res, data[j,:])
        centroids[i,:] = res / count
    return centroids

x = np.array([1, 2, 4])
y = np.array([2, 4, 1])
x[2] = 1

idx = findClosestCentroids(data, initial_centroids)
centroids = computeCentroids(data, idx, 3)


