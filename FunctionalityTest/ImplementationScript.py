from PIL import Image

import numpy as np
im = Image.open('bird_small.png')

pix = im.load()
data = np.asarray(im)
im1 = Image.fromarray(data)
im1.show()

norm = data / 255;
s = norm.shape

dim = s[0]*s[1];

fData = np.reshape(norm,(dim,s[2]))

def initCentroids(k, X):
    randix = np.random.permutation(fData.shape[0])
    randix = randix[0:k]
    centroids = X[randix, :]
    return centroids

def runKMeans(X, centroids, iters):


    return;

def findClosestCentroids(X, centroids):
    (k,d) = centroids.shape
    idx = np.zeros((k,1))
    for i in range(0,X.shape[0]):
        min = 1000000000000
        xr = X[i,:]
        best = 1
        for j in range(0,k):
            p = centroids[:,j]
            sub = np.subtract(p,xr)
            result = np.dot(sub,sub)
            if (result < min):
                best = j
                min = result
        idx[i,1] = best
    return idx

def computeCentroids(X, idx, K):
    (m,n) = X.shape
    centroids = np.zeros(K,n)


    return

x = np.array([1, 2, 4])
y = np.array([2, 4, 1])
x[2] = 1
print(x)

##print(initCentroids(10,fData))
