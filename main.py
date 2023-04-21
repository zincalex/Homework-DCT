import sys
import cv2 
import utilities as u



def main() :
    #taking from command line the img and the block size N
    img_file = sys.argv[1]
    N = int(sys.argv[2])
    imgBGR = cv2.imread("image input/" + img_file)

    R = 10
    R_LIMIT = 100
    
    #OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
    #converting BGR to RGB; once we want to show the image remember to convert it back
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCrCb) 
    Y, Cr, Cb = cv2.split(imgYCrCb)
    
    PSNR_values = []
    R_values = []

    while R <= R_LIMIT :
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
        

        #IF YOU LIKE TO SAVE THE FINAL IMAGE REBUILT, JUST UNCOMMENT THIS SECTION
        #imgYCrCb_rebuilt = cv2.merge([Y_rebuilt, Cr_rebuilt, Cb_rebuilt])
        #out_img = cv2.cvtColor(imgYCrCb_rebuilt, cv2.COLOR_YCrCb2BGR)
        #cv2.imwrite("Output image/" + str(R) + " compression rate.jpg", out_img) 
        
        MSE_Y = u.MSE(Y, Y_rebuilt)
        MSE_Cb = u.MSE(Cb, Cb_rebuilt)
        MSE_Cr = u.MSE(Cr, Cr_rebuilt)
        MSE_P = u.MSE_P(MSE_Y, MSE_Cb, MSE_Cr)
        PSNR = u.PNSR(MSE_P)
        
        R_values.append(R)
        PSNR_values.append(PSNR)
        R += 10
    
    u.PSNR_plot(R_values, PSNR_values, img_file, N)

if __name__ == "__main__" :
    main()