# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:18:32 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from joblib import dump, load

df = pd.read_csv(r'C:\portilla\ml_2022\bases\Advertising.csv')


X = df.drop('sales', axis=1)
y = df['sales']

polynominal_converter = PolynomialFeatures(degree=2,
                                           include_bias=False)

polynominal_converter.fit(X)

poly_features = polynominal_converter.transform(X)

#polynominal_converter.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

df['sales'].mean()

MAE = mean_absolute_error(y_test, test_predictions)
RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

model.coef_

'''
loop
'''

train_rmse_errors = []
test_rmse_errors = []

for d in range(1, 10):
    
    poly_converter = PolynomialFeatures(degree=d,
                                        include_bias=False)
    
    poly_features = poly_converter.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rsme = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rsme = np.sqrt(mean_squared_error(y_test, test_pred))
    
    
    train_rmse_errors.append(train_rsme)
    test_rmse_errors.append(test_rsme)
    
plt.plot(range(1, 6), train_rmse_errors[:5],
         label='TRAIN RSME ERROR')
plt.plot(range(1, 6), test_rmse_errors[:5],
         label='TEST RSME ERROR')
plt.ylabel('RMSE')
plt.xlabel('DEGREE OF POLY')

final_poly_converter = (
    PolynomialFeatures(degree=3,
                       include_bias=False)
)

final_model = LinearRegression()

full_converted_X = final_poly_converter.fit_transform(X)
final_model.fit(full_converted_X, y)

dump(final_model, r'C:\portilla\ml_2022\modelos\pr_sales.joblib')
loaded_model = load(r'C:\portilla\ml_2022\modelos\pr_sales.joblib')

dump(final_poly_converter, r'C:\portilla\ml_2022\modelos\pr_converter.joblib')
loaded_converter = load(r'C:\portilla\ml_2022\modelos\pr_converter.joblib')

campaign = [[149, 22, 12]]
campaign = loaded_converter.fit_transform(campaign)

loaded_model.predict(campaign)

