import os
import glob
import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


def mask2points(mask):
    mask = binary_fill_holes(mask).astype('uint8')
    ret,thresh = cv2.threshold(mask*255,127,255,0)

    contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pt_msk = np.zeros(thresh.shape[:2],dtype='uint8')
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv2.circle(pt_msk, (cx, cy), pt_size, (255, 255, 255), -1)
            pt_msk[cy,cx]=1

    return pt_msk

