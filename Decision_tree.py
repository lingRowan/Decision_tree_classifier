#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:11:08 2024

@author: rowanahmed
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score



data = pd.read_csv('/Users/rowanahmed/anaconda3/pkgs/bokeh-3.2.1-py311hb6e6a13_0/lib/python3.11/site-packages/bokeh/sampledata/_data/iris.csv')
data.head(20)


data.describe()

data.isna().sum()
data.dtypes


features = data.loc[:, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = data.loc[:, 'species']
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
model_DTC =  DecisionTreeClassifier()

model_DTC.fit(features_train, target_train)
predictions = model_DTC.predict(features_test)

accuracy = accuracy_score(target_test, predictions)
F1 = f1_score(target_test, predictions, average='weighted')
recall = recall_score(target_test, predictions, average='weighted')
conf_matrix = confusion_matrix(target_test, predictions)

print('f1', F1)
print('accuracy', accuracy)
print('recall', recall)
print('conf_matrix', conf_matrix)

plt.figure(figsize=(12, 8))  
plot_tree(model_DTC, 
           feature_names=features.columns.tolist(), 
           class_names=model_DTC.classes_.tolist(), 
           filled=True, 
           rounded=True, 
           fontsize=12)
plt.title("Decision Tree Visualization")
plt.show()


