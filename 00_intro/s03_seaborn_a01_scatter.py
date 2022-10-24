# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:42:53 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\dm_office_sales.csv')

plt.figure(figsize=(12, 4))
sns.scatterplot(x='salary', 
                y='sales', 
                hue='level of education',
                palette='Dark2',
                data=df)



plt.figure(figsize=(12, 4))
sns.scatterplot(x='salary', 
                y='sales', 
                size='salary',
                data=df)


plt.figure(figsize=(12, 4))
sns.scatterplot(x='salary', 
                y='sales', 
                s=150,
                alpha=0.3,
                data=df)


plt.figure(figsize=(12, 4))
sns.scatterplot(x='salary', 
                y='sales', 
                style='level of education',
                hue='level of education',
                data=df)
