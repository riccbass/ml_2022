# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:00:26 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import classification_report, plot_confusion_matrix

import seaborn as sns
sns.set(style='darkgrid')

df = pd.read_csv(r'C:\portilla\ml_2022\bases\penguins_size.csv')

df['species'].value_counts()

df.isnull().sum()

df = df.dropna()

sns.pairplot(df[df['species'] == 'Gentoo'], hue='sex') #fill bad data for single row sex

#female, 

df['sex'] = (

    np.where(df['sex'] == '.',
             'FEMALE',
             df['sex'])    
    
)

df['sex'].value_counts()

sns.pairplot(df, hue='species') #fill bad data for single row sex


X = pd.get_dummies(df.drop(['species'], axis=1),
                   drop_first=True)

y = df['species']

'''
no need for scaling in decision tree
'''

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

model = DecisionTreeClassifier()
model.fit(X_train, y_train) 

base_preds = model.predict(X_test)

print(classification_report(y_test, base_preds))

plot_confusion_matrix(model, X_test, y_test)

model.feature_importances_

df_features = pd.DataFrame(index=X_test.columns, data=model.feature_importances_)

'''
visualize tree
'''

plt.figure(figsize=(15, 8), dpi=200)
plot_tree(model, 
          feature_names=X.columns,
          filled=True)

def report_model(model):
    
    model_preds = model.predict(X_test)
    print(classification_report(y_test, model_preds))
    print('\n')
    plt.figure(figsize=(15, 8), dpi=200)
    plot_tree(model, 
              feature_names=X.columns,
              filled=True)
    

report_model(model)

pruned_tree = DecisionTreeClassifier(max_depth=2)

pruned_tree.fit(X_train, y_train)

report_model(pruned_tree)

max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
max_leaf_tree.fit(X_train, y_train)
report_model(max_leaf_tree)


entropy_tree = DecisionTreeClassifier(criterion='entropy')
entropy_tree.fit(X_train, y_train)
report_model(entropy_tree)
