# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:18:29 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\mushrooms.csv')

X = df.drop('class', axis=1)
y = df['class']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

param_grid = {'n_estimators':[50, 100], #default 100
              'learning_rate': [0.1, 0.05, 0.2], #default 0.1
              'max_depth': [3, 4, 5]} #default 3
              
gb_model = GradientBoostingClassifier()

grid = GridSearchCV(gb_model, param_grid)

grid.fit(X_train, y_train)

preds = grid.predict(X_test)

grid.best_params_

print(classification_report(y_test, preds))

imports = grid.best_estimator_.feature_importances_
imports = pd.DataFrame(index=X_train.columns, data=imports, columns=['import'])

imports.sort_values(['import'], inplace=True)

imports = imports[imports['import'] > 0.05]

plt.figure(figsize=(14,6))
sns.barplot(x=imports.index,
            y=imports['import'])
plt.xticks(rotation=90)



