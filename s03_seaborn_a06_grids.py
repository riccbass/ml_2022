# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:13:36 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\StudentsPerformance.csv')
sns.set(style='darkgrid')

sns.catplot(data=df,
            x='gender',
            y='math score',
            kind='box',
            row='lunch')


sns.catplot(data=df,
            x='gender',
            y='math score',
            kind='box',
            col='lunch')

'''
PAIR GRID
'''

g = sns.PairGrid(data=df, hue='gender')

g = g.map_upper(sns.scatterplot)
g = g.map_diag(sns.histplot)
g = g.map_lower(sns.kdeplot)