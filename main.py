import numpy as np
import cv2 
import sys

#img_file = sys.arg[1]
sus = "img_test4.jpg"
imgBGR = cv2.imread("image input/" + sus)

#   OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
#   converting BGR to RGB; once we want to show the image remember to convert it back
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCR_CB) 
Y, Cr, Cb = cv2.split(imgYCrCb)

cv2.imshow("BGR image", imgBGR) 
cv2.imshow('Y', Y) 
cv2.imshow('Cr', Cr) 
cv2.imshow('Cb', Cb) 
#cv2.imwrite('Y.png', Y) 
#cv2.imwrite('onlyCb_as_bgr.png', onlyCb_as_bgr) 
#cv2.imwrite('onlyCr_as_bgr.png', onlyCr_as_bgr)

cv2.waitKey(0) 
cv2.destroyAllWindows()