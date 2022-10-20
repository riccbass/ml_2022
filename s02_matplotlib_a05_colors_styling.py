# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:37:36 2022

@author: ricar
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 11)

fig = plt.figure()

ax = fig.add_axes([0, 0, 1,1])

ax.plot(x, x, color='#a742f5', label='1') #RGB
ax.plot(x, x + 1, color='yellow', label='2') #RGB

ax.legend()

fig = plt.figure()

ax = fig.add_axes([0, 0, 1,1])

ax.plot(x, x, color='#a742f5', lw=10, ls='--') #RGB
ax.plot(x, x + 1, color='yellow', linewidth=2, linestyle='-.') #RGB



fig = plt.figure()

ax = fig.add_axes([0, 0, 1,1])

ax.plot(x, x, color='yellow', lw=2, marker='o') #RGB
ax.plot(x, x + 1, color='red', lw=2, marker='+', ms=20) #RGB




