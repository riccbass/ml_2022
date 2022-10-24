# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:54:02 2022

@author: ricar
"""

import numpy as np

arr = np.arange(0, 11)

arr[8]

arr[0:5]

arr[:5]

arr[5:]

'''
broadcast
'''

arr[0:5] = 100


arr = np.arange(0, 11)

'''
slice
'''

slice_arr = arr[0:5] 

slice_arr[:] = 99 #mudou também o arr

arr = np.arange(0, 11)

slice_arr = arr[0:5].copy()

slice_arr[:] = 99 #mudou também o arr


'''
slice 2d array
'''

arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])

arr_2d.shape

arr_2d[1][2]

arr_2d[:2, 1]
arr_2d[:2, 1:]

'''
conditional selecting
'''

arr = np.arange(1, 11)

arr[arr > 4]







