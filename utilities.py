import numpy as np
import cv2 
import math

def my_dct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    block_list = matrix_to_blocks(np.float32(matrix), N, height, lenght)
    
    #Applying the dct transformation per block
    dct_block_list = []
    for elem in block_list : 
        dct_block_list.append(cv2.dct(elem))

    #Building back the whole matrix
    return blocks_to_matrix(dct_block_list, N, height, lenght)



def my_idct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    block_list = matrix_to_blocks(np.float32(matrix), N, height, lenght) 
    
    #Applying the dct INVERTED transformation per block
    idct_block_list = []
    for mtrx in block_list : 
        idct_block_list.append(cv2.idct(mtrx))

    #Building back the whole matrix, but returned the 8 bit unsigned integer version so the matrix can be immediatly used
    return np.uint8(np.round(blocks_to_matrix(idct_block_list, N, height, lenght)))



def matrix_to_blocks(matrix, N, h, l) : 
    fix_mtrx = padding(matrix, N, h, l)
    new_h, new_l = fix_mtrx.shape
    block_list = []

    for row_block in  range(0, math.ceil(new_h/N)) :
        for col_block in range(0, math.ceil(new_l/N)) : 
                
                col_indx = np.minimum(new_l - col_block*N, N)
                row_indx = np.minimum(new_h - row_block*N, N)

                block = np.zeros((row_indx, col_indx))
                for i in range(row_indx) :
                    for j in range(col_indx) : 
                        block[i][j] = fix_mtrx[row_block*N + i][col_block*N + j] 
                block_list.append(block)

    return block_list



def blocks_to_matrix(blocklist, N, mtrx_h, mtrx_l) : 
    build_mtrx = np.zeros((mtrx_h, mtrx_l), np.float32)
    numblock_in_row = math.ceil(mtrx_l / N)
    
    for i in range(0, mtrx_h) :
        for j in range(0, mtrx_l) : 
            build_mtrx[i][j] = blocklist[math.floor(i / N) * numblock_in_row + math.floor(j / N)][i%N][j%N]
                         
    return build_mtrx


def padding(matrix, N, h, l) : 
    row_rem = h % N
    col_rem = l % N
    return np.pad(matrix, pad_width = ((0, 0 if row_rem % 2 == 0 else 1), (0, 0 if col_rem % 2 == 0 else 1)), mode = 'edge')



def MSE (og_mtrx, compressed_mtrx) :
    height, lenght = og_mtrx.shape
    sque = np.float64(0)
    for i in range(0, height) :
        for j in range(0, lenght) :
            sque += (int(compressed_mtrx[i][j]) - int(og_mtrx[i][j]))**2
    return sque / (height * lenght)



def MSE_P(MSE_Y, MSE_Cb, MSE_Cr) : return 0.75*MSE_Y + 0.125*MSE_Cb + 0.125*MSE_Cr



def PNSR(MSE_P) : return (10 * math.log((255**2 / MSE_P), 10)) if MSE_P != 0 else 0



#TODO REMEMBER TO DELETE THIS METHOD, JUST FOR TESTING PURPOSE
def scale(matrix, mult):
    height, width = matrix.shape
    ret = np.zeros((height*mult, width*mult))
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, mult):
                for c in range(0,mult):
                    ret[i*mult+k][j*mult+c] = matrix[i][j]

    return ret
