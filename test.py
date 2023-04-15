import cv2
import utilities as u 
import numpy as np

N = 4
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
#cv2.imshow('og', np.uint8(u.scale(test_faded_d_errors, scale)))
sus = u.my_dct(test_faded_d_errors, N)
#cv2.imshow('new', np.uint8(u.scale(sus, scale)))
sus_final = u.my_idct(sus, N)
print(sus_final)

m = u.MSE(test_faded_d_errors, sus_final)
print(m)
#cv2.imshow('new', np.uint8(u.scale(sus_final, scale)))