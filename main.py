import numpy as np
import cv2 
import random
import sys

N = (8,16,64)
R = random.randint(1,100) #range [1,100]

def my_dct(matrix, N) :
    block_matrix_maker(matrix, N)
    ret = 0
    #for based on N to apply DCT
    #Apply a %R to make values 0
    return ret 


def block_matrix_maker(matrix, N) : 
    fix_matrix = padding(matrix, N)
    new_h, new_l = fix_matrix.shape
    ret = []
    for row_block in  range(0, int(new_h/N)) :
        for col_block in range(0, int(new_l/N)) : 
            block = np.zeros((N, N))
            for i in range(0,N) :
                for j in range(0,N) : 
                    block[i][j] = fix_matrix[row_block*N + i ][col_block*N + j] #LOOK FOR POSSIBLE ONE LINE COMMAND
            ret.append(block)
    print(ret)
    return ret

def padding(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns

    #comments explaining the pad method
    n_based_matrix = np.pad(matrix, pad_width = ((0, (N - height % N) if height%N != 0 else 0), (0, (N - lenght % N) if lenght%N != 0 else 0)), mode = 'edge')
    return n_based_matrix







#img_file = sys.arg[1]
sus = "img_test4.jpg"
imgBGR = cv2.imread("image input/" + sus)

#OpenCV show the image based on the BGR convention. In order to make things simpler to read, we procede 
#converting BGR to RGB; once we want to show the image remember to convert it back
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2YCR_CB) 
Y, Cr, Cb = cv2.split(imgYCrCb)

# float conversion
Y_f = np.float32(Y) 
Cr_f = np.float32(Cr) 
Cb_f = np.float32(Cb) 

# find discrete cosine transform (DCT)
Y_dct = cv2.dct(Y_f, cv2.DCT_INVERSE)
Cr_dct = cv2.dct(Cr_f, cv2.DCT_INVERSE)
Cb_dct = cv2.dct(Cb_f, cv2.DCT_INVERSE)


mn = block_matrix_maker(Y_f, 8)









#cv2.imshow("BGR image", imgBGR) 
#cv2.imshow('Y', Y) 
#cv2.imshow('Cr', Cr) 
#cv2.imshow('Cb', Cb) 

#SAVE THE IMAGES
#cv2.imwrite('Y.png', Y) 
#cv2.imwrite('onlyCb_as_bgr.png', onlyCb_as_bgr) 
#cv2.imwrite('onlyCr_as_bgr.png', onlyCr_as_bgr)

cv2.waitKey(0) 
cv2.destroyAllWindows()