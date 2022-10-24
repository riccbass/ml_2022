# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:54:02 2022

@author: ricar
"""

import numpy as np


arr = np.arange(0, 10)

arr + 5

arr + arr #same shape


arr - arr #same shape

arr / arr #same shape, only warning, not full error

np.sqrt(arr)

np.sin(arr)

np.log(arr)

'''
stats
'''

arr.sum()


arr.mean()


arr.max()

arr.var()

arr.std()

'''
axes
'''

arr2d = np.arange(0, 25).reshape(5, 5)


arr2d.sum() #por default é global


arr2d.sum(axis=0, keepdims=True) #sum of rows
arr2d.sum(axis=1, keepdims=True) #sum of columns


