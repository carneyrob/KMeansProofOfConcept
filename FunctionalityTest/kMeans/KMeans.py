import numpy as np
import random as rd

class KMeans:
    def __init__(self, data):
        self.data = np.array(data)

    def _findClosestCentroids(self, centroids):
        (k, d) = centroids.shape
        (m, n) = self.data.shape
        idx = np.zeros((m, 1))
        for i in range(0, m):
            min = 1000000000000
            xr = self.data[i, :]
            best = 1
            for j in range(0, k):
                p = centroids[j, :]
                sub = np.subtract(p, xr)
                result = np.dot(sub, sub)
                if (result < min):
                    best = j
                    min = result
            idx[i, 0] = best
        return idx

    def _computeCentroids(self, idx, K):
        (m, n) = self.data.shape
        centroids = np.zeros((K, n))
        for i in range(0, K):
            countInt = (idx == i).sum().astype(int)
            count = np.zeros((1, n))
            res = np.zeros((1, n))
            for p in range(0, n):
                count[0, p] = countInt
            for j in range(0, m):
                if (idx[j, 0] == i):
                    res = np.add(res, self.data[j, :])
            centroids[i, :] = res / count
        return centroids

    def runKMeans(self, iters):
        centroids = self.randomCentroidInit(20)
        (m, n) = self.data.shape
        (K, l) = centroids.shape
        idx = np.zeros((m, 1))
        for i in range(0, iters):
            idx = self._findClosestCentroids(centroids)
            centroids = self._computeCentroids(idx, K)
        return (idx, centroids)

    def randomCentroidInit(self, numCent):
        count = 0
        usedInds = []
        centroids = np.ones((numCent, self.data.shape[1]))
        while (count < numCent):
            ind = rd.randint(0, self.data.shape[0])
            if (ind in usedInds):
                continue
            usedInds.append(ind)
            centroids[count,:] = self.data[count,:]
            count += 1
        return centroids