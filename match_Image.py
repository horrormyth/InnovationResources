__author__ = 'devndraghimire'

import numpy as np
import cv2


orb = cv2.ORB_create()
#video capture
capture = cv2.VideoCapture(0)
#start image processing
def_Image = cv2.imread('images.jpeg')
changeto_Gray = cv2.cvtColor(def_Image,cv2.COLOR_BGR2GRAY)
kp_image,des_image = orb.detectAndCompute(changeto_Gray,None)


#For video
while True:
    # Frame retrieving
    ret, vid_Image = capture.read()
    gray_Vid = cv2.cvtColor(vid_Image,cv2.COLOR_BGR2GRAY)

    # Keypoint and Descriptors
    kp_Vid, des_Vid = orb.detectAndCompute(gray_Vid,None)

    # matches by bruteforce matcher using knn algorithm
    bfmatcher = cv2.BFMatcher()
    allmatch = bfmatcher.knnMatch(des_image,des_Vid,k=2)

    # put the matches in Array having the match distance satisfied by 75 %
    best_Match =[]
    for matcha, matchb in allmatch:
        if matcha.distance < 0.75 * matchb.distance:
            best_Match.append([matcha])

    # Traces the matches only, not the best match -- tracing best matches

    # TO-DO

    finalimage= cv2.drawMatchesKnn(def_Image,kp_image,gray_Vid,kp_Vid,allmatch,None,flags=2)
    cv2.imshow('win',finalimage)






