# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 08:58:37 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\StudentsPerformance.csv')
sns.set(style='darkgrid')

sns.jointplot(data=df,
              x='math score',
              alpha=1,
              y='reading score')


sns.jointplot(data=df,
              x='math score',
              y='reading score',
              kind='hex')

sns.jointplot(data=df,
              x='math score',
              y='reading score',
              kind='hist')


sns.jointplot(data=df,
              x='math score',
              y='reading score',
              shade=True,
              kind='kde')

sns.jointplot(data=df,
              x='math score',
              hue='gender',
              alpha=1,
              y='reading score')

'''
pairplot
'''

sns.pairplot(data=df,
             hue='gender',
             diag_kind='hist')

sns.pairplot(data=df,
             hue='gender',
             corner=True,
             diag_kind='hist')