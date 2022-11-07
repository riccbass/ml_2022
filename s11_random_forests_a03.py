# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:46:18 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\rock_density_xray.csv')

df.columns = ['Signal', 'Density']

sns.scatterplot(x='Signal', y='Density', data=df)

X = df['Signal'].values.reshape(-1, 1) #necessary for 1 column X

y = df['Density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

mean_absolute_error(y_test, lr_preds)
np.sqrt(mean_squared_error(y_test, lr_preds))

signal_range = np.arange(0, 100).reshape(-1, 1)
signal_preds = lr_model.predict(signal_range)

plt.figure(figsize=(12, 8), dpi=200)
sns.scatterplot(x='Signal', y='Density', data=df)

plt.plot(signal_preds)

'''
poly
'''

def run_model(model, X_train, X_test, y_train, y_test):
    
    #fit
    
    model.fit(X_train, y_train)
    
    #get metrics
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f'MAE: {mae}')    
    print(f'RMSE: {rmse}')
    
    
    #plot
    
    signal_range = np.arange(0, 100).reshape(-1, 1)
    signal_preds = model.predict(signal_range)

    plt.figure(figsize=(12, 8), dpi=200)
    sns.scatterplot(x='Signal', y='Density', data=df, color='black')

    plt.plot(signal_preds)



#model = LinearRegression()
#run_model(model, X_train, X_test, y_train, y_test)



'''
poly pipe
'''

pipe = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
run_model(pipe, X_train, X_test, y_train, y_test)


'''
others algorithms
'''

'''
knee
'''

k_values = [1, 5, 10]

for k in k_values:
    
    model = KNeighborsRegressor(n_neighbors=k)
    run_model(model, X_train, X_test, y_train, y_test)
    
'''
decision tree
'''
    
model = DecisionTreeRegressor()
run_model(model, X_train, X_test, y_train, y_test)
    

'''
svm
'''
svr = SVR()

param_grid = {'C':[0.01, 0.1, 1, 5, 10, 100, 1_000],
              'gamma':['auto', 'scale']}

grid = GridSearchCV(svr, param_grid)

run_model(grid, X_train, X_test, y_train, y_test)

'''
rf
'''

rfr = RandomForestRegressor(n_estimators=100)

run_model(rfr, X_train, X_test, y_train, y_test)

'''
boost
'''

model = GradientBoostingRegressor()
run_model(model, X_train, X_test, y_train, y_test)


model = AdaBoostRegressor()
run_model(model, X_train, X_test, y_train, y_test)





