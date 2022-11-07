# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:33:23 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')

X = df.drop(['sales'], axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #final holdout

scaler = StandardScaler()
scaler.fit(X_train) #X_train only to avoid data lekage

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


base_elastic_model = ElasticNet(tol=0.0001)

param_grid = {'alpha':[0.1, 1, 5, 10, 50, 100], 
              'l1_ratio':[.1, .5, .7, .95, .99, 1]}


grid_model = GridSearchCV(estimator=base_elastic_model, 
                          param_grid=param_grid, 
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=2)

grid_model.fit(X_train, y_train)


grid_model.best_estimator_

df_results = pd.DataFrame(grid_model.cv_results_)

y_pred = grid_model.predict(X_test)

mean_squared_error(y_test, y_pred)
