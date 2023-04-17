import sys
import random
import cv2 
import numpy as np
import utilities as u

def main() :
    #img_file = sys.arg[1]
    R = random.randint(1,100) #range [1,100]
    N = 128
    sus = "img_test4.jpg"
    imgBGR = cv2.imread("image input/" + sus)

    #OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
    #converting BGR to RGB; once we want to show the image remember to convert it back
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCrCb) 
    Y, Cr, Cb = cv2.split(imgYCrCb)
    
    first = True
    PSNR_values = []
    R_values = []

    while R <= 100 :

        Y_dct = u.my_dct(Y, N)
        Cr_dct = u.my_dct(Cr, N)
        Cb_dct = u.my_dct(Cb,N)
        
        #cv2.imshow('YDCT', Y_dct) 
        #cv2.imshow('CbDCT', Cb_dct) 
        #cv2.imshow('CrDCT', Cr_dct) 

        Y_dct_comp = u.percentage_loss(Y_dct, R)
        Cr_dct_comp = u.percentage_loss(Cr_dct, R)
        Cb_dct_comp = u.percentage_loss(Cb_dct, R)
        
        Y_rebuilt = u.my_idct(Y_dct_comp, N)
        Cr_rebuilt = u.my_idct(Cr_dct_comp, N)
        Cb_rebuilt = u.my_idct(Cb_dct_comp, N)
        

        imgYCrCb_rebuilt = cv2.merge([Y_rebuilt, Cr_rebuilt, Cb_rebuilt])
        out = cv2.cvtColor(imgYCrCb_rebuilt, cv2.COLOR_YCrCb2BGR)
        file_name = str(R) + '.jpg'
        cv2.imwrite(file_name, out) 
        

        MSE_Y = u.MSE(Y, Y_rebuilt)
        MSE_Cb = u.MSE(Cb, Cb_rebuilt)
        MSE_Cr = u.MSE(Cr, Cr_rebuilt)
        MSE_P = u.MSE_P(MSE_Y, MSE_Cb, MSE_Cr)
        PSNR = u.PNSR(MSE_P)

        print(R)
        print(MSE_Y)
        print(MSE_Cb)
        print(MSE_Cr)
        print(MSE_P)
        print(PSNR)
        print("NEXT \n")
        
        if not first : 
            R_values.append(R)
            PSNR_values.append(PSNR)
            R += 10
        else : 
            R = 10
            first = False

        
        
        #cv2.imshow('Cb rebuilt', Cb_rebuilt) 
        #cv2.imshow('Cr rebuilt', Cr_rebuilt) 
        

        #SAVE THE IMAGES
       
        #cv2.imwrite('onlyCb_as_bgr.png', onlyCb_as_bgr) 
        #cv2.imwrite('onlyCr_as_bgr.png', onlyCr_as_bgr)


if __name__ == "__main__" :
    main()