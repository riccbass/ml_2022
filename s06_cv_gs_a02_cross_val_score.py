# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:33:23 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')

X = df.drop(['sales'], axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #final holdout

scaler = StandardScaler()
scaler.fit(X_train) #X_train only to avoid data lekage

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)


scores = cross_val_score(model, X_train, y_train, 
                         scoring='neg_mean_squared_error',
                         cv=5)

model.fit(X_train, y_train)

abs(scores.mean())


model = Ridge(alpha=1)

scores = cross_val_score(model, X_train, y_train, 
                         scoring='neg_mean_squared_error',
                         cv=5)

abs(scores.mean())

model.fit(X_train, y_train)

y_final_pred = model.predict(X_test)

mean_squared_error(y_test, y_final_pred)