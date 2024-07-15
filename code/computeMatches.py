import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#


def SSD(img1,img2):
    ssd=np.sum(np.square(img1-img2),axis=1)
    # f,s =np.partition(ssd, 1)[0:2]
    return ssd

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """
    n=f1.shape[0]
    matches=np.zeros((n,))
    for i in range(n):
        f1Repeat=np.repeat(f1[i][np.newaxis,:],repeats=f2.shape[0],axis=0)
        ssd = SSD(f1Repeat,f2)
        ssdSort = np.sort(ssd)
        ind = np.argsort(ssd)
        if(ssdSort[0]/ssdSort[1]>=0.8):
            matches[i]=-1
        else:
            matches[i]=int(ind[0])
    return matches
