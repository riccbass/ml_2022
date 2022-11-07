# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:26:23 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score

import seaborn as sns


df = pd.read_csv(r'C:\portilla\ml_2022\bases\data_banknote_authentication.csv')

df['Class'].value_counts()

sns.pairplot(df, hue='Class')

X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

GridSearchCV()

n_estimators = [64, 100, 128, 200]
max_features = [2, 3, 4]

bootstrap = [True, False]
oob_score = [True, False]

param_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'bootstrap':bootstrap,
              'oob_score':oob_score}


rfc = RandomForestClassifier()
grid = GridSearchCV(rfc, param_grid)

grid.fit(X_train, y_train)

grid.best_params_

rfc = RandomForestClassifier(max_features=2, 
                             n_estimators=128,
                             bootstrap=True, 
                             oob_score=True)

rfc.fit(X_train, y_train)

rfc.oob_score_


predictions = rfc.predict(X_test)

plot_confusion_matrix(rfc, X_test, y_test)

print(classification_report(y_test, predictions))


errors = []
missclassifications = []

for n in range(1, 200):
    
    rfc = RandomForestClassifier(max_features=2, 
                                 n_estimators=n,
                                 bootstrap=True, 
                                 oob_score=True)
    
    rfc.fit(X_train, y_train)
    preds = rfc.predict(X_test)
    
    err = 1 - accuracy_score(y_test, preds)
    
    n_missesd = np.sum(preds != y_test)
    
    errors.append(err)
    missclassifications.append(n_missesd)
    
plt.plot(range(1, 200,), errors)


plt.plot(range(1, 200,), missclassifications)
    
