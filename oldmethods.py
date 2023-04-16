def my_dct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    pad_l = lenght - ((lenght % N - N) if lenght%N != 0 else 0)
    block_list = matrix_to_blocks(np.float32(matrix), N, height, lenght)
    
    #Applying the dct transformation per block
    dct_block_list = []
    for mtrx in block_list : 
        dct_block_list.append(cv2.dct(mtrx))

    #Building back the whole matrix
    return blocks_to_matrix(dct_block_list, height, lenght, pad_l, N)



def my_idct(matrix, N) :
    height, lenght = matrix.shape  #number rows, number columns
    pad_l = lenght - ((lenght % N - N) if lenght%N != 0 else 0)
    block_list = matrix_to_blocks(np.float32(matrix), N, height, lenght) 
    
    #Applying the dct INVERTED transformation per block
    idct_block_list = []
    for mtrx in block_list : 
        idct_block_list.append(cv2.idct(mtrx))

    #Building back the whole matrix, but returned the 8 bit unsigned integer version so the matrix can be immediatly used
    return np.uint8(blocks_to_matrix(idct_block_list, height, lenght, pad_l, N))



def blocks_to_matrix(blocklist, mtrx_h, mtrx_l, pad_l, N) : 
    build_mtrx = np.zeros((mtrx_h, mtrx_l))
    for i in range(0, mtrx_h) :
        for j in range(0, mtrx_l) : 
            build_mtrx[i][j] = blocklist[int((i - i%N) / N) * int(pad_l/N) + int((j - j%N) / N)][i%N][j%N]

    return build_mtrx



def matrix_to_blocks(matrix, N, h, l) : 
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