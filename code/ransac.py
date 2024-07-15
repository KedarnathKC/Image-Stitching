import numpy as np
import cv2
import time

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

np.random.seed(1)
def findAT(A,B):
    return np.linalg.lstsq(A,B)[0]
    
def getInlier(A,B,M,threshold):
    inliers=list()
    BTransform=np.dot(A,M)
    distance = BTransform-B
    distance = np.square(distance)
    distance = distance.reshape((A.shape[0]//2,2))
    distance = np.sum(distance,axis=1)
    distance = np.sqrt(distance)
    inliers = np.argwhere(distance<threshold)
    return inliers.reshape((-1))
    
def prepAB(samples,blobs1,blobs2,pairs):
    n = samples.shape[0]
    A=list()
    B=list()
    for k in range(n):
        srcX,srcY = blobs1[pairs[samples[k],1],0],blobs1[pairs[samples[k],1],1]
        destX,destY = blobs2[pairs[samples[k],0],0],blobs2[pairs[samples[k],0],1]
        A.append([srcX,srcY,0,0,1,0])
        A.append([0,0,srcX,srcY,0,1])
        B.append(destX)
        B.append(destY)
    A=np.array(A)
    B=np.array(B)   
    return A,B

def ransac(matches, blobs1, blobs2):
    m = np.indices(matches.shape).reshape(matches.shape)
    pairs = np.stack((m, matches), axis=1) 
    pairs = pairs[pairs[:, 1] != -1]
    n = pairs.shape[0]
    pairs=pairs.astype(int)
    maxItr=16
    maxInliers = 0
    best_transf = np.zeros((2,3))
    T = np.zeros(6,)
    inliers_idx = []
    threshold=5
    for i in range(maxItr):
        samples = np.random.randint(0,n,size=3)
        A,B = prepAB(samples,blobs2,blobs1,pairs)
        X = findAT(A,B)
        A,B = prepAB(np.indices((pairs.shape[0],))[0],blobs2,blobs1,pairs)
        inliers = getInlier(A,B,X,threshold)
        if inliers.shape[0] > maxInliers:
            maxInliers = inliers.shape[0]
            A,B = prepAB(samples,blobs2,blobs1,pairs)
            T = findAT(A,B)
            inliers_idx = inliers
        best_transf[0][0] = T[0]
        best_transf[0][1] = T[1]
        best_transf[0][2] = T[4]
        best_transf[1][0] = T[2]
        best_transf[1][1] = T[3]
        best_transf[1][2] = T[5]
    return inliers_idx,best_transf