# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:22:06 2022

@author: ricar
"""

'''
simple linear regression (one feature only)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')

df['total_spend'] = df[['TV', 'radio', 'newspaper']].sum(axis=1)

sns.scatterplot(data=df,
                x='total_spend',
                y='sales')

sns.regplot(data=df,
                x='total_spend',
                y='sales')

X = df['total_spend']
y = df['sales']

#y = mx + b
help(np.polyfit)

np.polyfit(X, y, deg=1)

potential_spend = np.linspace(0, 500, 100)
predict_sales = 0.04868788*potential_spend + 4.24302822


sns.scatterplot(data=df,
                x='total_spend',
                y='sales')

plt.plot(potential_spend, predict_sales, color='red')

spend = 200
predict_sales = 0.04868788*spend + 4.24302822

#y = B3x**3 + B2x**1 + B1 + B0
np.polyfit(X, y, deg=3)


potential_spend = np.linspace(0, 500, 100)
predict_sales = 3.07615033e-07*potential_spend**3 + -1.89392449e-04*potential_spend**2 \
               + 8.20886302e-02*potential_spend +  2.70495053e+00


sns.scatterplot(data=df,
                x='total_spend',
                y='sales')

plt.plot(potential_spend, predict_sales, color='red')
    