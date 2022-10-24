# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 06:34:25 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\dm_office_sales.csv')
sns.set(style='darkgrid')

df['division'].value_counts()

plt.figure(figsize=(10, 4))
sns.countplot(data=df,x='division')

plt.figure(figsize=(10, 4))
sns.countplot(data=df,x='division')
plt.ylim(50)


plt.figure(figsize=(10, 4))
sns.countplot(data=df,x='level of education', hue='division',
              palette='Set2')


'''
barplot
'''

plt.figure(figsize=(10, 4))
sns.barplot(data=df, 
            x='level of education', 
            y='salary',
            ci='sd',
            estimator=np.mean)


plt.figure(figsize=(10, 4))
sns.barplot(data=df, 
            x='level of education', 
            y='salary',
            ci='sd',
            hue='division',
            estimator=np.mean)
plt.legend(bbox_to_anchor=(1.05, .5))