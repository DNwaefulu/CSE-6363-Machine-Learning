# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:00:18 2023

@author: dnwae
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import pandas as pd
import seaborn as sns
import pickle

data = pd.read_csv('iris.csv')
sns_plot = sns.pairplot(data)
sns_plot.figure.savefig("iris_output.jpg")

models = {'model5': {'x':['petal_length','petal_width','sepal_length'],'y':'sepal_width'},
        'model6': {'x':['petal_length','petal_width'],'y':'sepal_width'},
        'model7': {'x':['sepal_length'],'y':'sepal_width'},
        'model8': {'x':['petal_length'],'y':'petal_width'}}

results = {'model5':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model6':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model7':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model8':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]}}

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(data[['sepal_length','sepal_width','petal_length','petal_width']],data[['species']],stratify=data[['species']],test_size=0.10)

for model in models:
    print("__________ {} _____________".format(model))
    X_train, X_test, y_train, y_test = X_train_split[models[model]['x']], X_test_split[models[model]['x']], X_train_split[models[model]['y']], X_test_split[models[model]['y']]
    l = LinearRegression()
    MSE = l.fit(X_train, y_train)
    results[model]['validation_mse']=l.value_losses
    results[model]['train_mse']=l.train_losses
    results[model]['weights']=l.weights
    l.predict(X_test)
    test_mse = l.score(X_test, y_test)
    print("\nScore method MSE= ",test_mse)
    print("\n")
    
    for n in range(MSE[1]):
        print("Validation Mean Squared Error: {} for step {}".format(MSE[0], n))
        print('\n')