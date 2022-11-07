# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 07:40:24 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'C:\portilla\ml_2022\bases\gene_expression.csv')


sns.countplot(data=df,
            x='Cancer Present')

sns.pairplot(df,
             hue='Cancer Present')

X = df.drop(['Cancer Present'], axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train) #avoid data leakeage
scaled_X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(scaled_X_train, y_train)

y_pred = knn_model.predict(scaled_X_test)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

'''
elbow method
'''

test_error_rates = []

for k in range(1, 30):
    
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train, y_train)

    y_pred = knn_model.predict(scaled_X_test)

    test_error = 1 - accuracy_score(y_test, y_pred)
    
    test_error_rates.append(test_error)
    

plt.plot(range(1, 30),
         test_error_rates)
plt.ylabel('ERROR RATE')
plt.xlabel('K Neighbors')

'''
PIPELINE --> GRIDSEARCH CV
'''

scaler = StandardScaler()
knn = KNeighborsClassifier()

knn.get_params().keys()

operations = [('scaler', scaler), ('knn', knn)]

pipe = Pipeline(operations)

k_values = list(range(1, 20))

param_grid = {'knn__n_neighbors':k_values}

full_cv_classifier = GridSearchCV(pipe, 
                                  param_grid,
                                  cv=5,
                                  scoring='accuracy')

full_cv_classifier.fit(X_train, y_train) #no need for scaling, pipe does it

full_cv_classifier.best_estimator_.get_params()

full_pred = full_cv_classifier.predict(X_test) #no need for scaling, pipe does it

print(classification_report(y_test, full_pred))

new_patient = [[3.8, 6.4]]

full_cv_classifier.predict(new_patient)

full_cv_classifier.predict_proba(new_patient)

