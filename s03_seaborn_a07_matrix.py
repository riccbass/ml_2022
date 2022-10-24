# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:13:36 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\country_table.csv')
sns.set(style='darkgrid')

df = df.set_index('Countries')

plt.figure(figsize=(15, 8))
sns.heatmap(df.drop(['Life expectancy'], axis=1),
            linewidth=0.5,
            cmap='viridis',
            annot=True,
            center=40)

plt.figure(figsize=(15, 8))
sns.clustermap(df.drop(['Life expectancy'], axis=1),
               linewidth=0.5,
               cmap='viridis',
               annot=True,
               col_cluster=False,
               center=40)