# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 08:24:35 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.ensemble import AdaBoostClassifier

import seaborn as sns

df = pd.read_csv(r'C:\portilla\ml_2022\bases\mushrooms.csv')

sns.countplot(data=df,
              x='class')

desc_df = df.describe().T.reset_index().sort_values(by='unique')

plt.figure(figsize=(14,6))
sns.barplot(data=desc_df,
            x='index',
            y='unique')
plt.xticks(rotation=90)

X = df.drop('class', axis=1)

X.isnull().sum()

X = pd.get_dummies(X, drop_first=True)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

model = AdaBoostClassifier(n_estimators=1)

'''
only best feature to advise
'''

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))


X_train.columns[model.feature_importances_.argmax()]
    
sns.countplot(data=df,
              x='odor',
              hue='class')


error_rates = []

for n in range(1, 96):
        
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    err = 1 - accuracy_score(y_test, preds)
    
    error_rates.append(err)

plt.plot(range(1, 96),
         error_rates)

model

feats = pd.DataFrame(index=X.columns,
                     data=model.feature_importances_,
                     columns=['Importance'])

imp_feats = feats[feats['Importance'] > 0]

imp_feats = imp_feats.sort_values(by='Importance')

plt.figure(figsize=(14,6))
sns.barplot(data=imp_feats,
            x=imp_feats.index,
            y=imp_feats.Importance)
plt.xticks(rotation=90)