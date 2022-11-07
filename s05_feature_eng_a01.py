# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:45:38 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# Choose a mean,standard deviation, and number of samples

def create_ages(mu=50,sigma=13,num_samples=100,seed=42):

    # Set a random seed in the same cell as the random call to get the same values as us
    # We set seed to 42 (42 is an arbitrary choice from Hitchhiker's Guide to the Galaxy)
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages,decimals=0)
    
    return sample_ages

sample = create_ages()

sns.displot(data=sample,
            bins=20)

sns.boxplot(sample)

ser = pd.Series(sample)

ser.describe()

IQR = 55.25 - 42
lower_limit = 42 - 1.5*IQR

ser[ser > lower_limit]

'''
using np to calculate iqr
'''

np.percentile(sample, [75])
q75, q25 = np.percentile(sample, [75, 25])
iqr = q75 - q25

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Ames_Housing_Data.csv')

corr_df = df.corr()['SalePrice']

sns.scatterplot(data=df, 
                x='Overall Qual',
                y='SalePrice')


sns.scatterplot(data=df, 
                x='Gr Liv Area',
                y='SalePrice')

mask1 = df['Overall Qual'] >= 9
mask2 = df['SalePrice'] < 200_000

possible_ol = df[mask1 & mask2]

mask1 = df['Gr Liv Area'] >= 4_000
mask2 = df['SalePrice'] < 400_000

possible_ol = df[mask1 & mask2]

df = df[~mask1 | ~mask2]

sns.scatterplot(data=df, 
                x='Gr Liv Area',
                y='SalePrice')


df.to_csv(r'C:\portilla\ml_2022\bases\Ames_without_outliers.csv')


