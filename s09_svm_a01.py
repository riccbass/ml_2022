# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:32:35 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='darkgrid')

import sys
sys.path.insert(0, r'C:\portilla\ml_2022')

from sklearn.svm import SVC

df = pd.read_csv(r'C:\portilla\ml_2022\bases\mouse_viral_study.csv')

sns.countplot(data=df,
            x='Virus Present')

sns.scatterplot(data=df,
               x='Med_1_mL',
               y='Med_2_mL',
               hue='Virus Present')

#HYPERPLANE

x = np.linspace(0, 10, 100)
m = -1
b = 11
y = m*x + b

plt.plot(x, 
         y,
         'black')

y = df['Virus Present']
X = df.drop(['Virus Present'], axis=1)

model = SVC(kernel='linear',
            C=1_000)

model.fit(X, y)

from svm_margin_plot import plot_svm_boundary


plot_svm_boundary(model, X, y)

