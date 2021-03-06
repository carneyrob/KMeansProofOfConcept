import numpy as np
from PIL import Image
import random as rd
import warnings
import time as tm


print('Running')
# For loading images as NP arrays
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data.astype(np.float32)

# For saving NP arrays as images (given a file location)
def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
#
#  Base K-Means Functionality
#
def runKMeans(X, centroids, iters):
    vFunc = np.vectorize(findClosestCentroid, excluded=['centroids'])
    (m,n) = X.shape
    (K,l) = centroids.shape
    idx = np.zeros((m,1))
    for i in range(0,iters):
        t0 = tm.time()
        idx = vFunc(X,centroids=centroids)
        print("Time 1 {0}: {1}".format(i, tm.time() - t0))
        centroids = computeCentroids(X,idx,K)
        t1 = tm.time() - t0
        print("Time 2 {0}: {1}".format(i,t1))
    return (idx,centroids)

def findClosestCentroid(row, centroids):
    (k, d) = centroids.shape
    min = 100000000;
    best = 1
    for i in range(0,k):
        p = centroids[i,:]
        sub = np.subtract(p,row)
        result = sub.dot(sub)
        if (result < min):
            best = 1
            min = result
    return best



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

def computeCentroids(X, idx, K):
    (m,n) = X.shape
    centroids = np.zeros((K,n))
    count1 = 0
    count2 = 0
    for i in range(0,K):
        count1 += 1
        countInt = (idx == i).sum().astype(int)
        count = np.zeros((1,n))
        res = np.zeros((1, n))
        for p in range(0,n):
            count[0,p] = countInt
        for j in range(0,m):
           if (idx[j,0]==i):
                res = np.add(res, X[j,:])
        if countInt == 0:
            count2 += 1
            centroids[i,:] = getRandomCentroid(X)
        else:
            centroids[i,:] = res / count
    return centroids

def getRandomCentroid(X):
    ind = rd.randint(0,X.shape[0])
    return X[ind,:]

def randomCentInit(dim, numCent, low=0, high=100):
    return np.random.randint(low, high, (numCent, dim)).astype(np.float32)

def img_reshape(img):
    try:
        shape = img.shape
        return np.reshape(img, (shape[0]*shape[1], shape[2]))
    except IndexError:
        print("Cannot reshape array ith shape: {0}".format(img.shape))
    except Exception as err:
        print("Unexpected error")

init_img = load_image('../bird_uncompressed.png')

test_img = img_reshape(init_img)

init_centroids = randomCentInit(test_img.shape[1], 150, 0, 255)
(idx, colors) = runKMeans(test_img, init_centroids, 25)
idx = findClosestCentroids(test_img, colors)

#indTran = np.transpose(idx)[0].astype(int)
#img_rec = colors[indTran]
#img_shaped = np.reshape(img_rec, init_img.shape).astype(np.uint8)
#img_res = Image.fromarray(img_shaped)
#img_res.save('compressed_img.png',"PNG")







