"""
def OLD_blocks_to_matrix(blocklist, mtrx_h, mtrx_l, pad_l, N) : 
    build_mtrx = np.zeros((mtrx_h, mtrx_l))
    for i in range(0, mtrx_h) :
        for j in range(0, mtrx_l) : 
            build_mtrx[i][j] = blocklist[int((i - i%N) / N) * int(pad_l/N) + int((j - j%N) / N)][i%N][j%N]
    return build_mtrx
"""



"""
def OLD_matrix_to_blocks(matrix, N, h, l) : 
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
"""



"""
def OLDpadding(matrix, N, h, l) :
    Nbased_mtrx = np.pad(matrix, pad_width = ((0, (N - h % N) if h%N != 0 else 0), (0, (N - l % N) if l%N != 0 else 0)), mode = 'edge')
    return Nbased_mtrx
"""