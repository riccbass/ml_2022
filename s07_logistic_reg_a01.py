# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:36:59 2022

@author: ricar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve


df = pd.read_csv(r'C:\portilla\ml_2022\bases\hearing_test.csv')

df.describe()

X = df.drop(['sales'], axis=1)
y = df['sales']

df['test_result'].value_counts()

sns.countplot(df['test_result'])

sns.boxplot(x='test_result',
            y='age',
            data=df)


sns.boxplot(x='test_result',
            y='physical_score',
            data=df)

sns.scatterplot(x='age',
                y='physical_score',
                data=df)

plt.figure(dpi=150)
sns.scatterplot(x='age',
                y='physical_score',
                hue='test_result',
                alpha=0.4,
                data=df)

#always use that firstly classificiation problem
sns.pairplot(data=df,
             hue='test_result')


sns.heatmap(df.corr(),
        annot=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(df['age'], 
           df['physical_score'], 
           df['test_result'],
           c=df['test_result'])

plt.show()


X = df.drop(['test_result'], axis=1)
y = df['test_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(scaled_X_train, y_train)

log_model.coef_

y_pred = log_model.predict(scaled_X_test)
y_pred_proba = log_model.predict_proba(scaled_X_test)

accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(log_model, scaled_X_test, y_test)

print(classification_report(y_test, y_pred))

plot_roc_curve(log_model, scaled_X_test, y_test)

plot_precision_recall_curve(log_model, scaled_X_test, y_test)


