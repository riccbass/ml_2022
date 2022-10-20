# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:54:02 2022

@author: ricar
"""

import numpy as np

my_list = [1, 2, 3]

my_array = np.array(my_list)

'''
2d - matrix
'''

my_matrix = [[1,2, 3],
             [4,5, 6],
             [7, 8, 9]]

np.array(my_matrix)


'''
build
'''

np.arange(0, 10, )

np.arange(0, 10, 2)

'''
np zeros
'''

np.zeros(5)


np.zeros((2, 5)) #rows first, column second


'''
np ones
'''


np.ones((2, 5)) #rows first, column second


'''
linspace
'''

#not same arange

np.linspace(0, 15, 3) #default, endpoind included

'''
random distrubeted
'''

np.random.rand(2, 10 ) #vai shape ou quantiadde

'''
random normal distrubeted
'''

np.random.randn(2, 10 ) #vai shape ou quantiadde

'''
random integral
'''

np.random.randint(0, 101, (3, 5) ) #vai intervalo e shape


np.random.randint(0, 101, 10 ) #vai intervalo e quantidade

'''
define o seed
'''

np.random.seed(42)
np.random.rand(4 ) #vai quantiadde

'''
atributos e métodos
'''

arr = np.arange(0, 25)

arr.reshape(5, 5) #só funciona se tiver mesma quantidade, no caso 5x5 = 25

ranarr = np.random.randint(0, 101, 10)
ranarr.max()
ranarr.min()
ranarr.argmax() #índice 5 tem o maior
ranarr.argmin() #índice 7 tem o menor

ranarr.dtype





