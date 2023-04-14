import sys
import random
import cv2 
import numpy as np
import utilities as u

N = 64
R = random.randint(1,100) #range [1,100]

def main() :
    #img_file = sys.arg[1]
    sus = "img_test4.jpg"
    imgBGR = cv2.imread("image input/" + sus)

    #OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
    #converting BGR to RGB; once we want to show the image remember to convert it back
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCR_CB) 
    Y, Cr, Cb = cv2.split(imgYCrCb)

    #cv2.imshow('RGB', imgBGR)
    #cv2.imshow('Y', Y) 
    cv2.imshow('Cb', Cb) 
    #cv2.imshow('Cr', Cr) 
    

    Y_dct = u.my_dct(Y, N)
    Cr_dct = u.my_dct(Cr, N)/255
    Cb_dct = u.my_dct(Cb,N)/255
    
    #cv2.imshow('YDCT', Y_dct) 
    cv2.imshow('CbDCT', Cb_dct) 
    #cv2.imshow('CrDCT', Cr_dct) 
    
    Y_idct = u.my_idct(Y_dct, N)
    Cb_idct = u.my_idct(Cb_dct,N)
    Cr_idct = u.my_idct(Cr_dct, N)

    # convert to uint8
    #Y_rebuilt = np.uint8(Y_idct)
    Cb_rebuilt = np.uint8(Cb_idct)
    Cr_rebuilt = np.uint8(Cr_idct)

    #cv2.imshow('Y rebuilt', Y_rebuilt) 
    cv2.imshow('Cb rebuilt', Cb_rebuilt) 
    #cv2.imshow('Cr rebuilt', Cr_rebuilt) 
    

    #SAVE THE IMAGES
    #cv2.imwrite('Y.png', Y) 
    #cv2.imwrite('onlyCb_as_bgr.png', onlyCb_as_bgr) 
    #cv2.imwrite('onlyCr_as_bgr.png', onlyCr_as_bgr)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()