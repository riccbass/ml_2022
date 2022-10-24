# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 06:52:43 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\StudentsPerformance.csv')
sns.set(style='darkgrid')

plt.figure(figsize=(15, 10))
sns.boxplot(data=df,
            y='math score',
            x='parental level of education',
            hue='test preparation course',
            palette='Set2')
plt.legend(bbox_to_anchor=(1.05, 0.5))

'''
violin
'''

plt.figure(figsize=(15, 10))
sns.violinplot(data=df,
               y='reading score',
               hue='test preparation course',
               x='parental level of education',
               palette='Set2')
plt.legend(bbox_to_anchor=(1.05, 0.5))


plt.figure(figsize=(15, 10))
sns.violinplot(data=df,
               y='reading score',
               inner=None,
               hue='test preparation course',
               x='parental level of education',
               palette='Set2')
plt.legend(bbox_to_anchor=(1.05, 0.5))



plt.figure(figsize=(15, 10))
sns.violinplot(data=df,
               y='reading score',
               inner='quartile',
               hue='test preparation course',
               x='parental level of education',
               palette='Set2')
plt.legend(bbox_to_anchor=(1.05, 0.5))


'''
violin noise
'''

plt.figure(figsize=(15, 10))
sns.violinplot(data=df,
               y='reading score',
               inner='quartile',
               bw=0.01,
               hue='test preparation course',
               x='parental level of education',
               palette='Set2')
plt.legend(bbox_to_anchor=(1.05, 0.5))


'''
swarm
'''

sns.swarmplot(data=df, x='math score', size=2)


sns.swarmplot(data=df, 
              x='math score', 
              size=2,
              y='gender',
              hue='test preparation course')
plt.legend(bbox_to_anchor=(1.05, 0.5))


sns.swarmplot(data=df, 
              x='math score', 
              size=2,
              dodge=True,
              y='gender',
              hue='test preparation course')
plt.legend(bbox_to_anchor=(1.05, 0.5))

