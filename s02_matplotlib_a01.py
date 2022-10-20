# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:09:32 2022

@author: ricar
"""

import matplotlib.pyplot as plt

import numpy as np

x = np.arange(0, 10)
y = 2 * x

plt.plot(x, y)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.xlim(0,6)
plt.ylim(0,15)
plt.title('String Title')

