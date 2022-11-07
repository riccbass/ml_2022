# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:00:55 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

with open(r'C:\portilla\ml_2022\bases\Ames_Housing_Feature_Description.txt', 'r') as f:
    description = f.read()

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Ames_no_missing.csv')

df['Lot Frontage'].fillna(df['Lot Frontage'].mean(), inplace=True)

df.isnull().sum().sum()

df['MS SubClass'] = df['MS SubClass'].apply(str)

my_object_df = df.select_dtypes(include='object')
my_numeric_df = df.select_dtypes(exclude='object')


df_object_dummies = pd.get_dummies(my_object_df,
                                   drop_first=True)

final_df = pd.concat([my_numeric_df, df_object_dummies],
                     axis=1)

final_df.corr()['SalePrice'].sort_values()