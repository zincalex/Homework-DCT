import sys
import random
import cv2 
import numpy as np
import utilities as u

N = 256
R = random.randint(1,100) #range [1,100]

def main() :
    #img_file = sys.arg[1]
    sus = "img_test1.jpg"
    imgBGR = cv2.imread("image input/" + sus)

    #OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
    #converting BGR to RGB; once we want to show the image remember to convert it back
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCrCb) 
    Y, Cr, Cb = cv2.split(imgYCrCb)

    cv2.imshow('RGB', imgBGR)
    #cv2.imshow('Y', Y) 
    #cv2.imshow('Cb', Cb) 
    #cv2.imshow('Cr', Cr) 
    
    Y_dct = u.my_dct(Y, N)
    Cr_dct = u.my_dct(Cr, N)
    Cb_dct = u.my_dct(Cb,N)
    
    #cv2.imshow('YDCT', Y_dct) 
    #cv2.imshow('CbDCT', Cb_dct) 
    #cv2.imshow('CrDCT', Cr_dct) 

    #TODO ADD THIS %R FACTOR DONT KNOW WHAT TO DO YET
    
    Y_rebuilt = u.my_idct(Y_dct, N)
    Cb_rebuilt = u.my_idct(Cb_dct,N)
    Cr_rebuilt = u.my_idct(Cr_dct, N)

    imgYCrCb_rebuilt = cv2.merge([Y_rebuilt, Cr_rebuilt, Cb_rebuilt])
    out = cv2.cvtColor(imgYCrCb_rebuilt, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('asdfa', out)

    MSE_Y = u.MSE(Y, Y_rebuilt)
    MSE_Cb = u.MSE(Cb, Cb_rebuilt)
    MSE_Cr = u.MSE(Cr, Cr_rebuilt)
    MSE_P = u.MSE_P(MSE_Y, MSE_Cb, MSE_Cr)
    PSNR = u.PNSR(MSE_P)


    #TODO ITERATE THIS FOR FIRST TIME WITH RANDOM R, THEN BY INCREASING R FROM 10 TO 100
    # IN 10 STEPS 
    

    #cv2.imshow('Y rebuilt', Y_rebuilt) 
    #cv2.imshow('Cb rebuilt', Cb_rebuilt) 
    #cv2.imshow('Cr rebuilt', Cr_rebuilt) 
    

    #SAVE THE IMAGES
    #cv2.imwrite('Y.png', Y) 
    #cv2.imwrite('onlyCb_as_bgr.png', onlyCb_as_bgr) 
    #cv2.imwrite('onlyCr_as_bgr.png', onlyCr_as_bgr)


    """
    test_faded_d_errors = np.array([
    [255, 140, 216, 198, 180, 162, 144, 126],
    [234, 216, 198, 180, 165, 144, 126, 108],
    [216, 198, 170, 162, 144, 126, 108, 90],
    [198, 180, 162, 144, 126, 108, 100, 72],
    [180, 158, 144, 100, 108, 90, 72, 54],
    [162, 144, 126, 108, 90, 72, 54, 18],
    [144, 126, 108, 88, 72, 54, 0, 18],
    [126, 108, 90, 72, 54, 36, 18, 0,]
    ])

    scale = 50
    cv2.imshow('og', np.uint8(u.scale(test_faded_d_errors, scale)))
    sus = u.my_dct(test_faded_d_errors, N)
    #cv2.imshow('new', np.uint8(u.scale(sus, scale)))
    sus_final = u.my_idct(sus, N)
    print(sus_final)
    cv2.imshow('new', np.uint8(u.scale(sus_final, scale)))
    """

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()