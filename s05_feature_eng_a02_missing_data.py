# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:45:38 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

with open(r'C:\portilla\ml_2022\bases\Ames_Housing_Feature_Description.txt', 'r') as f:
    print(f.read())

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Ames_without_outliers.csv')


df.drop(['PID'], axis=1, inplace=True)
df.drop([df.columns[0]], axis=1, inplace=True) #remove previous saved index

missing = df.isnull().sum() / len(df)
missing = missing[missing > 0]

missing = missing.sort_values()

plt.figure(figsize=(10, 5))
sns.barplot(x=missing.index,
            y=missing)
plt.xticks(rotation=90)
plt.ylim(0, 0.01)  

missing[missing < 0.01]

m_ele = df[pd.isnull(df['Electrical'])]

df.dropna(axis=0, 
          subset=['Electrical', 'Garage Cars'], 
          inplace=True)

missing = df.isnull().sum() / len(df)
missing = missing[missing > 0]

missing = missing.sort_values()

m_basem = df[pd.isnull(df['Bsmt Unf SF'])]

#bst numeric columns
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)


#bst string columns
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

missing = df.isnull().sum() / len(df)
missing = missing[missing > 0]

missing = missing.sort_values()

'''
masonry
'''

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)


missing = df.isnull().sum() / len(df)
missing = missing[missing > 0]

missing = missing.sort_values()


plt.figure(figsize=(10, 5))
sns.barplot(x=missing.index,
            y=missing)
plt.xticks(rotation=90)

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

df['Fireplace Qu'].fillna('None', inplace=True)


df['Lot Frontage N'] = (
    df.groupby(['Neighborhood'])['Lot Frontage'].transform('mean')
)

df['Lot Frontage'] = df['Lot Frontage N'] 

df.drop(['Lot Frontage N'], axis=1, inplace=True)

missing = df.isnull().sum() / len(df)
missing = missing[missing > 0]

missing = missing.sort_values()

df['Lot Frontage'].fillna(df['Lot Frontage'].mean(), inplace=True)

df.to_csv(r'C:\portilla\ml_2022\bases\Ames_no_missing.csv',
                 index=False)

