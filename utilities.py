import numpy as np
import cv2 
import math
import matplotlib.pyplot as plt

def my_dct(matrix, N) :
    """
    This function break up the matrix in blocks and then apply to each of them 
    the default dct provided by opencv 

    Args:
        matrix : Matrix to which apply the dct
        N : The block dimension

    Returns:
        The dct matrix rebuild from the blocks
    """

    height, width = matrix.shape  
    block_list = matrix_to_blocks(np.float32(matrix), N, height, width)
    
    #Applying the dct transformation per block
    dct_block_list = []
    for elem in block_list : 
        dct_block_list.append(cv2.dct(elem))

    return blocks_to_matrix(dct_block_list, N, height, width)



def my_idct(matrix, N) :
    """
    This function break up the matrix in blocks and then apply to each of them 
    the default inversed dct provided by opencv. Then the matrix is build back from
    the block list

    Args:
        matrix : Matrix to which is applied the idct
        N : The block dimension

    Returns:
        The idct matrix rebuild from the blocks
    """

    height, width = matrix.shape  
    block_list = matrix_to_blocks(np.float32(matrix), N, height, width) 
    
    #Applying the dct INVERTED transformation per block
    idct_block_list = []
    for mtrx in block_list : 
        idct_block_list.append(cv2.idct(mtrx))
    idct_mtrx = np.round(blocks_to_matrix(idct_block_list, N, height, width))
    
    #some value might be higher than 255 or less than 0 after the idct, resulting the conversation in int8 to broke some image parts
    for i in range(height) : 
        for j in range (width) : 
            if idct_mtrx[i][j] > 255 : idct_mtrx[i][j] = 255 
            if idct_mtrx[i][j] <  0 : idct_mtrx[i][j] = 0
    
    #Building back the whole matrix, however the 8 bit unsigned integer version is returned; so the matrix can be immediatly used
    return np.uint8(idct_mtrx)



def matrix_to_blocks(matrix, N, h, w) : 
    """
    Given a matrix, this function break up the matrix in blocks. The dimension of the blocks are generally
    NxN, however on the edges of the matrix if the height and/or the width do not fit with NxN blocks, the dimension
    of the matrix is temporally changed to have blocks even sized (e.g with a 7x9 matrix and N=4, starting from the top-left 
    corner, only 2 4x4 blocks can fit; since the number of columns and rows is odd, we temporally change the matrix dimension
    to 8x10. In this way the "third" block in the list will have a 4x2 dimension. The first row is ended, we move to the fifth row
    and keep iterate). The columns and rows must be even because the opencv dct accepts only even sized matrixes/blocks

    Args:
        matrix : Matrix to be decomposed
        N : The block dimension
        h : The height of the matrix
        w : The width of the matrix

    Returns:
        A list with all blocks
    """

    fix_mtrx = padding(matrix, N, h, w) #changing size
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



def blocks_to_matrix(blocklist, N, mtrx_h, mtrx_w) : 
    """
    This function build from a given list of blocks a matrix, from which the block where derived

    Args:
        blocklist : A list of blocks
        N : The block dimension
        mtrx_h : The height of the matrix
        mtrx_w : The width of the matrix

    Returns:
        The builded matrix
    """

    build_mtrx = np.zeros((mtrx_h, mtrx_w), np.float32)
    numblock_in_row = math.ceil(mtrx_w / N)
    
    for i in range(0, mtrx_h) :
        for j in range(0, mtrx_w) : 
            build_mtrx[i][j] = blocklist[math.floor(i / N) * numblock_in_row + math.floor(j / N)][i%N][j%N]
                         
    return build_mtrx


def padding(matrix, N, h, w) : 
    """
    This function resize a given matrix to its even form. If a row and/or column is added, the new values
    added next to the edge are copied from the near row/column

    Args:
        matrix : Matrix to be resized
        N : The block dimension
        h : The height of the matrix
        w : The width of the matrix

    Returns:
        The even sized matrix
    """

    row_rem = h % N
    col_rem = w % N
    return np.pad(matrix, pad_width = ((0, 0 if row_rem % 2 == 0 else 1), (0, 0 if col_rem % 2 == 0 else 1)), mode = 'edge')


def percentage_loss (dct_mtrx, R) :
    """
    This function set to 0 the R% values of the given matrix

    Args:
        dct_matrix : A matrix to which the dct was applied
        R : The percentage loss

    Returns:
        The "compressed" version of the matrix
    """

    value = np.percentile(np.abs(dct_mtrx), R) #search the value for which %R in np.abs(dct_mtrx) are lower than this value returned
    mtrx_comp = np.copy(dct_mtrx)

    #The elements in the matrix lower than value are set to 0
    mtrx_comp[ np.abs(mtrx_comp) <= value ] = 0 #inline condition 

    return mtrx_comp


def MSE (og_mtrx, compressed_mtrx) :
    """
    Calculate the mean squared error between 2 matrixes

    Args:
        og_mtrx : original matrix
        compressed_mtrx : new matrix after some operations

    Returns:
        The mean squared error
    """

    height, width = og_mtrx.shape
    sque = np.float64(0)
    for i in range(0, height) :
        for j in range(0, width) :
            sque += (int(compressed_mtrx[i][j]) - int(og_mtrx[i][j]))**2
    return sque / (height * width)



#Weighted mean squared error
def MSE_P(MSE_Y, MSE_Cb, MSE_Cr) : return 0.75*MSE_Y + 0.125*MSE_Cb + 0.125*MSE_Cr



def PNSR(MSE_P) : return (10 * math.log((255**2 / MSE_P), 10)) if MSE_P != 0 else np.inf



def PSNR_plot(R_values, PSNR_values, img_file, N) :
    """
    Simple function to save a PSNR plot in the ./Plot image folder

    Args: 
        R_values : x axis values
        PSNR_values : y axis values
        img_file : file name + extension
        N : block dimension
    Returns:
        None
    """
    plot = plt.figure()
    
    plt.title(img_file + " " + str(N) + "x" + str(N) + " blocks")
    plt.yscale("linear")
    plt.xlabel("R")
    plt.xticks(R_values)
    plt.ylabel("PSNR")
    plt.plot(R_values, PSNR_values, c = 'green')
    plt.grid()
    #plt.show()

    name_img = img_file.split('.')[0]
    plot.savefig("Plot image/Plot " + name_img + " " + str(N) + " blocks.jpg", bbox_inches = 'tight') 










