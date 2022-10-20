# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:09:32 2022

@author: ricar
"""

import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a ** 4

x = np.arange(0, 10)
y = 2*x

fig = plt.figure()
#LARGE AXES
axes1 = fig.add_axes([0, 0, 1, 1])
axes1.plot(a, b)
axes1.set_xlim(0,8)
axes1.set_ylim(0,8000)
axes1.set_xlabel('A')
axes1.set_ylabel('B')
axes1.set_title('Power of 4')
#SMALL AXES
axes2 = fig.add_axes([0.2, 0.2, 0.5, 0.5])
axes2.plot(x, y)
axes2.set_xlim(1,2)
axes2.set_ylim(0,50)
axes2.set_xlabel('A')
axes2.set_ylabel('B')
axes2.set_title('Zoomed in')
plt.show()

fig = plt.figure(figsize=  (12, 8), dpi = 100)

axes1 = fig.add_axes([0, 0, 1, 1])
axes1.plot(a, b)
