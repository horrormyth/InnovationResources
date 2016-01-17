import cv2
#static image file
myimg = cv2.imread('sprite.jpg')


#Initialize SURF object
orb = cv2.ORB_create()
imgrey = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
kp_myimg, des_myimage = orb.detectAndCompute(imgrey,None)
for km in kp_myimg:
    kx = int(km.pt[0])
    ky = int(km.pt[0])
    print('This is %d kx',kx)

print len(kp_myimg)

#Set desired radius
rad = 2
#Create object to read images from camera 0
cam = cv2.VideoCapture(0)
# cam.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 800)
# cam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, 600)


while True:
    #Get image from webcam and convert to greyscale
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect keypoints and descriptors in greyscale image
    keypoints, descriptors = orb.detectAndCompute(gray,None)
    #print len(keypoints)


#
    #Draw a small red circle with the desired radius
    #at the (x, y) location for each feature found
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        print('This is %d x',x)
        # print y
        cv2.circle(img, (x, y), rad, (0, 0, 255))

    #Display colour image with detected features
    cv2.imshow("features", img)

    #Sleep infinite loop for ~10ms
    #Exit if user presses <Esc>
    if cv2.waitKey(10) == 27:
        break
