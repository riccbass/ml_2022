# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:12:30 2022

@author: ricar
"""


import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a ** 4

x = np.arange(0, 10)
y = 2*x

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0][0].plot(x, y)
axes[1][1].plot(a, b)

plt.tight_layout()


fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0][0].plot(x, y)
axes[0][1].plot(x, y)
axes[1][0].plot(x, y)
axes[1][0].set_ylabel('Y LABEL 1,0')
axes[1][0].set_xlim(2,6)
axes[1][1].plot(a, b)

fig.subplots_adjust(wspace=1, hspace=1)
fig.suptitle('Super title')

plt.tight_layout()
