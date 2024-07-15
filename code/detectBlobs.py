import numpy as np
import cv2
from scipy.ndimage import correlate1d,generic_laplace,gaussian_laplace
from scipy.ndimage.filters import maximum_filter
from skimage.color import rgb2gray

# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn
#

def d2(input, axis, output, mode, cval):
    return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)

def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 5 array with blob in each row in (x, y, radius, angle, score)
    #
    # Dummy - returns a blob at the center of the image
    # blobs = np.array([[im.shape[1]/2, im.shape[0]/2, 100, 0, 1.0]])
    blobs=list()
    h,w,_=im.shape
    im=rgb2gray(im).astype(np.float64)
    n=param['n']
    k=param['k']
    threshold=param['threshold']
    sigma=param['sigma']
    Laplace=np.zeros((h,w,n))
    sigmas=list()
    radius=list()
    
    for i in range(n):
        Laplace[:,:,i]=(sigma)**2 *np.absolute(gaussian_laplace(im,sigma))
        sigmas.append(sigma)
        radius.append(sigma*1.414)
        sigma*=k
    
    NMSSpace=np.zeros(Laplace.shape)
    Laplace[Laplace<threshold]=0
    for i in range(n):
        NMSSpace[:, :, i] = maximum_filter(Laplace[:, :, i], size=3)
    indexs=np.argmax(NMSSpace,axis=2)
    NMSScale=np.max(NMSSpace,axis=2)
    NMS=NMSScale*(NMSScale==np.max(Laplace,axis=2))
    
    for x in range(h):
        for y in range(w):
            blob=[y,x,sigmas[indexs[x,y]]*np.sqrt(2),0,NMS[x,y]]
            blobs.append(blob)
    blobs=np.array(blobs)
    blobs=np.delete(blobs,np.where(blobs[:,4]<=0),axis=0)
    return blobs
    