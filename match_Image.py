__author__ = 'devndraghimire'

import numpy as np
import cv2
rad =2
orb = cv2.ORB_create()
capture = cv2.VideoCapture(0) #video capture
#stat image processing
def_Image = cv2.imread('dev.jpg')
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
        cv2.circle(gray_Vid,(x,y),rad,(0,0,255))

    #Find Matches
    bfmatcher = cv2.BFMatcher()
    allmatch = bfmatcher.knnMatch(des_Vid,des_image,k=2)
    # print allmatch
    # Test the matches
    best_Match =[]
    for matcha, matchb in allmatch:
        if matcha.distance < 0.75 * matchb.distance:
            best_Match.append(matcha)



#plotting
    # print best_Match
    # final_Match =cv2.drawMatches(def_Image,kp_image,vid_Image,des_Vid,best_Match,flags=2)
    # cv2.imshow('The MatchFrame',vid_Image)





