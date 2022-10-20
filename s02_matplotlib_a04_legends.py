# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:31:32 2022

@author: ricar
"""


import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = np.linspace(0, 10, 11)

ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, x, label='X vs X')
ax.plot(x, x ** 2, label='X vs X ** 2')

#ax.legend(loc='lower left')
ax.legend(loc=(1.1, 0.5))