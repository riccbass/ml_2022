# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:33:20 2022

@author: ricar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

import seaborn as sns
sns.set(style='darkgrid')

df = pd.read_csv(r'C:\portilla\ml_2022\bases\penguins_size.csv')

df = df.dropna()

df['sex'] = (

    np.where(df['sex'] == '.',
             'FEMALE',
             df['sex'])    
    
)

X = pd.get_dummies(df.drop(['species'], axis=1),
                   drop_first=True)

y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


rfc = RandomForestClassifier(10, 
                             max_features='auto',
                             random_state=101)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

plot_confusion_matrix(rfc, X_test, y_test)


print(classification_report(y_test, preds))


rfc.feature_importances_

X_train.columns


