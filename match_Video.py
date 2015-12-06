__author__ = 'devndraghimire'
__author__ = 'devndraghimire'

import numpy as np
import cv2

MIN_MATCH_COUNT =10

orb = cv2.ORB_create()
#video capture
capture = cv2.VideoCapture(0)
#start image processing
def_Image = cv2.imread('tert.jpg')
changeto_Gray = cv2.cvtColor(def_Image, cv2.COLOR_BGR2GRAY)
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
    good_Match =[]
    for matcha, matchb in allmatch:
        if matcha.distance < 0.75 * matchb.distance:
            good_Match.append(matcha)

    if len(good_Match)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp_image[matcha.queryIdx].pt for matcha in good_Match]).reshape(-1,1,2)
        des_pts = np.float32([kp_Vid[matcha.trainIdx].pt for matcha in good_Match]).reshape(-1,1,2)

        # Create a mask for the homography
        M, mask = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0)
        matched_Mask = mask.ravel().tolist()

        h,w = def_Image.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #Draw the perspective Transform box
        dst = cv2.perspectiveTransform(pts,M)

        # Draw lines on the video
        vid_Image = cv2.polylines(vid_Image,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        print '!!!!!!Matches Found!!!!!!- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
    #     The freet transform box will show only when the good matches are found

    else:
        print 'Not enough matches Found- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
        matched_Mask = None

    finalimage= cv2.drawMatchesKnn(def_Image,kp_image,vid_Image,kp_Vid,allmatch,None, flags=2)
    cv2.imshow('win',finalimage)

    if cv2.waitKey(10) == 27:
        break






