import numpy as np
from PIL import Image
import random as rd

# For loading images as NP arrays
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data.astype(np.float32) / 255

# For saving NP arrays as images (given a file location)
def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
#
#  Base K-Means Functionality
#
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
                res = np.add(res, X[j,:])
        centroids[i,:] = res / count
    return centroids

def randomCentInit(dim, numCent, low=0, high=100):
    return np.random.randint(low, high, (numCent, dim)).astype(np.float32)

def img_reshape(img):
    shape = img.shape;
    return np.reshape(img, (shape[0]*shape[1], shape[2]))

test = np.ones((3, 3, 2))
test[1, 1, 1] = 0
test[2, 2, 0] = 2
test[2, 1, 1] = 8
test[0, 1, 1] = 3
#test reshape here


test_img = img_reshape(load_image('../bird_uncompressed.png'))
print(test_img.shape)

init_centroids = randomCentInit(test_img.shape[1], 20, 0, 255)
print(init_centroids.shape)
(idx, colors) = runKMeans(test_img, init_centroids, 25)
print(idx.shape)





