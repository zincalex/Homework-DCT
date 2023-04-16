import cv2
import utilities as u 
import numpy as np
import math

def newPad(matrix, N, h, l) : #ZINCA SEAL
    row_rem = h % N
    col_rem = l % N
    return np.pad(matrix, pad_width = ((0, 0 if row_rem % 2 == 0 else 1), (0, 0 if col_rem % 2 == 0 else 1)), mode = 'edge')

def matrix_to_blocks(matrix, N, h, l) : #ZINCA SEAL
    fix_mtrx = newPad(matrix, N, h, l)
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


def blocks_to_matrix(blocklist, N, mtrx_h, mtrx_l) : #ZINCA SEAL
    build_mtrx = np.zeros((mtrx_h, mtrx_l), np.float32)
    numblock_in_row = math.ceil(mtrx_l / N)
    
    for i in range(0, mtrx_h) :
        for j in range(0, mtrx_l) : 
            build_mtrx[i][j] = blocklist[math.floor(i / N) * numblock_in_row + math.floor(j / N)][i%N][j%N]
                         
    return build_mtrx



