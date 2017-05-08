import numpy as np
from PIL import Image
import random as rd
import warnings
import time as tm


np.seterr(all='warn')
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
    (m,n) = X.shape
    (K,l) = centroids.shape
    idx = np.zeros((m,1))
    for i in range(0,iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,K)
    return (idx,centroids)

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



test_img = load_image('../bird_uncompressed.png')

img_shape = test_img.shape;

# TEST REGION
test = np.ones((3, 3, 2))
test[1, 1, 1] = 0
test[2, 2, 0] = 2
test[2, 1, 1] = 8
test[0, 1, 1] = 3
#END REGION

init_img = load_image('../test_img.png')

test_img = img_reshape(init_img)
#print(test_img.shape)

init_centroids = randomCentInit(test_img.shape[1], 20, 0, 255)
(idx, colors) = runKMeans(test_img, init_centroids, 25)
#print(idx.shape)

idx = findClosestCentroids(test_img, colors)
#print(idx[1:20])

indTran = np.transpose(idx)[0].astype(int)
img_rec = colors[indTran]
img_shaped = np.reshape(img_rec, init_img.shape).astype(np.uint8)
img_res = Image.fromarray(img_shaped)
img_res.save('compressed_img.png',"PNG")

test_idx = np.ones((10,1))
test_idx[1] = 0
test_idx[4] = 2

test_cent = np.ones((3, 2))
test_cent[1,:] = [2, 2]
test_cent[2,:] = [3, 3]

test_idx1 = [1, 2, 0]

test_res = test_cent[[1,2,0,1,1,1,0,2,1,2]]
#print(test_res)

test_tran = np.transpose([[1],[2],[3],[4]])
#print(test_tran[0])




#print(np.isfinite(test_img).all())
#print(np.isfinite(init_centroids).all())

#print(test_img[1:20][1:20][1:20])

