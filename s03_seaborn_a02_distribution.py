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
sns.set(style='darkgrid')

#plt.figure(figsize=(12, 4))
sns.rugplot(x='salary', 
            height=.5,
            data=df)


sns.displot(x='salary', 
            bins=10,
            data=df)



sns.displot(x='salary', 
            bins=10,
            edgecolor='blue',
            color='red',
            kde=True,
            data=df)


sns.histplot(data=df, 
             x='salary')

sns.kdeplot(data=df, 
             x='salary')

np.random.seed(42)
sample_ages = np.random.randint(0, 100, 200)

sample_ages = pd.DataFrame(sample_ages,
                           columns=['age'])

sns.rugplot(x='age',
            data=sample_ages)

sns.displot(x='age', 
            bins=30,
            rug=True,
            kde=True,
            data=sample_ages)

sns.kdeplot(data=sample_ages, 
            clip=[0, 100],
            bw_adjust=0.01,
            shade=True,
            x='age')