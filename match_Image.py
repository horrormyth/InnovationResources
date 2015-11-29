__author__ = 'devndraghimire'

import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

rad =2
orb = cv2.ORB_create()
capture = cv2.VideoCapture(0) #video capture
#stat image processing
def_Image = cv2.imread('Karhu.png')
changeto_Gray = cv2.cvtColor(def_Image,cv2.COLOR_BGR2GRAY)
kp_image,des_image = orb.detectAndCompute(changeto_Gray,None)
# print len(kp_image)
# print len(des_image)

#For video
while True:
    ret, vid_Image = capture.read()
    gray_Vid = cv2.cvtColor(vid_Image,cv2.COLOR_BGR2GRAY)
    kp_Vid, des_Vid = orb.detectAndCompute(gray_Vid,None)
    # print len(kp_Vid)
    for keypoints in kp_Vid:
        x=int(keypoints.pt[0])
        y=int(keypoints.pt[0])
        # cv2.circle(gray_Vid,(x,y),rad,(0,0,255))

    #Find Matches
    bfmatcher = cv2.BFMatcher()
    allmatch = bfmatcher.knnMatch(des_image,des_Vid,k=2)
    # print allmatch
    # Test the matches
    best_Match =[]
    for matcha, matchb in allmatch:
        if matcha.distance < 0.75 * matchb.distance:
            best_Match.append([matcha])
            # print len(best_Match)# length of the match

    # Traces the matches only, not the best match -- tracing best matches

    # TO-DO

    finalimage= cv2.drawMatchesKnn(changeto_Gray,kp_image,gray_Vid,kp_Vid,allmatch,None,flags=2)
    cv2.imshow('win',finalimage)






