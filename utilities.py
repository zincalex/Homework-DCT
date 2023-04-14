import numpy as np
import cv2 

def my_dct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    pad_l = lenght - ((lenght % N - N) if lenght%N != 0 else 0)
    block_list = block_matrix_maker(matrix, N, height, lenght)
    
    #Applying the dct transformation per block
    dct_block_list = []
    for mtrx in block_list : 
        dct_block_list.append(cv2.dct(mtrx))

    #Building back the whole matrix
    dct_mtrx = np.zeros((height, lenght))
    for i in range(0, height) :
        for j in range(0, lenght) : 
            dct_mtrx[i][j] = dct_block_list[int((i - i%N) / N) * int(pad_l/N) + int((j - j%N) / N)][i%N][j%N]
    return dct_mtrx



def my_idct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    pad_l = lenght - ((lenght % N - N) if lenght%N != 0 else 0)
    block_list = block_matrix_maker(matrix, N, height, lenght) 
    
    #Applying the dct INVERTED transformation per block
    idct_block_list = []
    for mtrx in block_list : 
        idct_block_list.append(cv2.idct(mtrx))

    #Building back the whole matrix 
    #TODO MIGHT CONSIDER CREATING A SINGLE METHOD FOR THIS PART SINCE IS THE SAME IN DCT
    idct_mtrx = np.zeros((height, lenght))
    for i in range(0, height) :
        for j in range(0, lenght) : 
            idct_mtrx[i][j] = idct_block_list[int((i - i%N) / N) * int(pad_l/N) + int((j - j%N) / N)][i%N][j%N]

    return idct_mtrx



def block_matrix_maker(matrix, N, h, l) : 
    fix_mtrx = padding(matrix, N, h, l)
    new_h, new_l = fix_mtrx.shape
    block_list = []

    for row_block in  range(0, int(new_h/N)) :
        for col_block in range(0, int(new_l/N)) : 
            block = np.zeros((N, N))
            for i in range(0,N) :
                for j in range(0,N) : 
                    block[i][j] = fix_mtrx[row_block*N + i][col_block*N + j] #LOOK FOR POSSIBLE ONE LINE COMMAND
            block_list.append(block)

    return block_list



#TODO EXPLAIN WHAT THIS METHOD DOES
def padding(matrix, N, h, l) :
    Nbased_mtrx = np.pad(matrix, pad_width = ((0, (N - h % N) if h%N != 0 else 0), (0, (N - l % N) if l%N != 0 else 0)), mode = 'edge')
    return Nbased_mtrx
