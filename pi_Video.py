__author__ = 'devndraghimire'

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

MIN_MATCH_COUNT =10
orb=cv2.ORB_create()
# (1) Image Reading and Converting start
def_image = cv2.imread('tert.jpg')
changeto_Gray = cv2.cvtColor(def_image,cv2.COLOR_BGR2GRAY)
kp_image, des_image = orb.detectAndCompute(changeto_Gray,None)
# end (1)
bfmatcher = cv2.BFMatcher()

#(2) pi Camera Initialize
capture = PiCamera()
capture.resolution = ()
capture.framerate =(1072,768)
capture.framerate = 32
raw_Capture = PiRGBArray(capture, size=(1024,768))

#(3) Frame Capture
for frame in capture.capture_continuous(
        raw_Capture,
        format='bgr',
        use_video_port=True
        ):
# (3) read video and convert
    vid = frame.array
    gray_Vid = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    kp_Vid, des_Vid = orb.detectAndCompute(gray_Vid,None)
    allmatch = bfmatcher.knnMatch(des_image,des_Vid,k=2)

# (4) 75 % of match
    good_Match =[]
    for matcha,matchb in allmatch:
        if matcha.distance < 0.75 *matchb.distance:
            good_Match.append(matcha)

    if len(good_Match)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp_image[matcha.queryIdx].pt for matcha in good_Match]).reshape(-1,2,2)
        des_pts = np.float32([kp_Vid[matcha.trainIdx].pt for matcha in good_Match]).reshape(-1,1,2)

        M,mask = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0)
        matched_Mask = mask.ravel().tolist()

        h,w = def_image.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

#(5) Draw lines in Video
        vid = cv2.polylines(vid,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        print '!!!!!!Matches Found!!!!!!- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
    else:
        print 'Not enough matches Found- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
        matched_Mask = None

    finalimage = cv2.drawMatchesKnn(def_image,kp_image,vid,kp_Vid,allmatch,None,flags=2)
    cv2.imshow('win',finalimage)
    if cv2.waitKey(10) == 27:
        break
