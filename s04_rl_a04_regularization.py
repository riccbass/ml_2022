# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:21:41 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, SCORERS
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from joblib import dump, load

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')

X = df.drop('sales', axis=1)
y = df['sales']

polynominal_converter = PolynomialFeatures(degree=3,
                                           include_bias=False)

poly_features = polynominal_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train) #use only train data to avoid data leakage
X_test = scaler.transform(X_test)

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train, y_train)

test_predictions = ridge_model.predict(X_test)

MAE = mean_absolute_error(y_test, test_predictions)
RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),
                         scoring='neg_mean_absolute_error')
ridge_cv_model.fit(X_train, y_train)

ridge_cv_model.alpha_

test_predictions = ridge_cv_model.predict(X_test)

MAE = mean_absolute_error(y_test, test_predictions) #diminuiu
RMSE = np.sqrt(mean_squared_error(y_test, test_predictions)) #diminuiu


ridge_cv_model.coef_


#SCORERS.keys()

'''
LASSO
'''


