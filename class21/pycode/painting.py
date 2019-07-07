#!/usr/bin/env python

import os
import cv2
import numpy as np

#step 1: load image, and set the known and unknown area
img = cv2.imread('./imgs/1.jpg')
os.makedirs("./out", exist_ok=True)
initmask = (img[:,:,2]>200) * (img[:,:,1]<100)
initmask = initmask.astype(np.uint8)*255
cv2.imwrite("./out/initmask.png", initmask)
confmap = np.zeros((initmask.shape), np.float32)
confmap[initmask==0] = 1.0

#step 2: iteratively fill the unknown area
iterid = 0
mask = initmask.copy()

#TODO: here we should iterate on just the pixels on the contour

#compute image gradient
# print(len(contour), contour[0].shape, contour[0][0,:,:])

while mask.sum()>0: #unknown area is not null

    


    #step 2.1: find the current contour
    contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 

    #step 2.2: find the fill-front pixel with max priority

    #precompute current confmap
    confmap_internal = cv2.boxFilter(confmap, -1, (9, 9))
    confmap_internal[initmask==0]=1.0

    for contour in contours:
        for cpixel in contour:
            #step 2.2.1: compute Cp, using the precomputed confmap as input. 
            #step 2.2.2: compute Dp
                #step 2.2.2.1
                #Compute the gradient value with maximum amplitude in the target patch
                #gradX=
                #gradY=
                #gradAmp=
                #choose the value with maximum gradAmp in the patch
                

                #step 2.2.2.2
                #Compute the normalize vector of cpixel direction
                
    
    
    



