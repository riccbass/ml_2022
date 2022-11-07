# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:07:59 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')

X = df.drop(['sales'], axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


'''
scale the data
'''

scaler = StandardScaler()
scaler.fit(X_train) #X_train only to avoid data lekage

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Ridge(alpha=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mean_squared_error(y_test, y_pred)

model_two = Ridge(alpha=1)
model_two.fit(X_train, y_train)

y_pred_two = model_two.predict(X_test)

mean_squared_error(y_test, y_pred_two)

'''
test, train, validation
'''

del X_test
del y_test

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)

X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train) #X_train only to avoid data lekage

X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

model_one = Ridge(alpha=100)
model_one.fit(X_train, y_train)

y_eval_pred = model_one.predict(X_eval)

mean_squared_error(y_eval, y_eval_pred)

model_two = Ridge(alpha=1)
model_two.fit(X_train, y_train)

new_pred_eval = model_two.predict(X_eval)

mean_squared_error(y_eval, new_pred_eval)

y_final_test_pred = model_two.predict(X_test)

mean_squared_error(y_test, y_final_test_pred)
