# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 01:12:26 2023

@author: dnwae
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas import get_dummies
from math import ceil
import csv
from LinearRegression import LinearRegression


# Load the dataset
data = pd.read_csv('vgsales.csv')  # Replace with your dataset file
print(data.isnull().sum())
data = data.dropna()
# print(data.head())
# print(data.columns)
# print(data.describe())
# print(data.info())

# sns.distplot(data['Global_Sales'])

print(data.corr())

# sns.heatmap(data.corr(), annot=True)

cols0 = ['Platform', 'Year', 'Genre', 'Publisher']

for col in cols0:
    chart = data[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()
    sns.set_style("white")
    plt.figure(figsize=(12.4, 5))
    plt.xticks(rotation=90)
    sns.barplot(x=col, y='Name', data=chart[:50], palette=sns.cubehelix_palette((12 if col == 'Genre' else 30), start = 2, dark=0.3, light=.85, reverse=True)).set_title(('Game count by '+col), fontsize=16)
    plt.ylabel('Count', fontsize=14)
    plt.xlabel('')
    
def Release_year(y_r):
    if y_r >= 2020:
        return '2020-2023'
    elif y_r >= 2015:
        return '2015-2019'
    elif y_r >= 2010:
        return '2010-2014'
    elif y_r >= 2005:
        return '2005-2009'
    elif y_r >= 2000:
        return '2000-2004'
    else:
        return '1980-1999'
# ########################### Year of game release ##############################
dfh = data.dropna(subset=['Year']).reset_index(drop=True)
dfh['Release_Years'] = data['Year'].apply(lambda x: Release_year(x))

def release_window(x):
    if x in pack:
        return x
    else:
        pass
def width(x):
    if x == 'Platform':
        return 14.4
    elif x == 'Publisher':
        return 11.3
    elif x == 'Genre':
        return 13.6

def height(x):
    if x == 'Genre':
        return 8
    else:
        return 9
    
cols1 = ['Platform', 'Genre', 'Publisher']

for col in cols1:
    pack = []
    top = dfh[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()[:15]
    for x in top[col]:
        pack.append(x)
    dfh[col] = dfh[col].apply(lambda x: release_window(x))
    dfh_platform = dfh[[col, 'Release_Years', 'Global_Sales']].groupby([col, 'Release_Years']).median().reset_index().pivot(col, "Release_Years", "Global_Sales")
    plt.figure(figsize=(width(col), height(col)))
    sns.heatmap(dfh_platform, annot=True, fmt=".2g", linewidths=.5).set_title((' \n'+col+' vs. release year (by median sales) \n'), fontsize=18)
    plt.ylabel('', fontsize=14)
    plt.xlabel('Release Year \n', fontsize=12)
    pack = []

df1 = data[['Year','Genre','Publisher','Global_Sales']]
df1 = df1.dropna().reset_index(drop=True)
# df1 = df1.astype('float64')

fig, ax = plt.subplots(1,1, figsize=(12,5))
sns.regplot(x="Year", y="Global_Sales", data=df1, ci=None, color="#75556c", x_jitter=.02).set(ylim=(0, 17.5))
sns.regplot(x="Year", y="Global_Sales", data=df1.loc[df1.Global_Sales >= 2.0], truncate=True, x_bins=15, color="#75556c", x_jitter=.02).set(ylim=(0, 17.5))

# Select relevant features and target
X = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
y = data['Global_Sales']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model = LinearRegression(batch_size=32, regularization=0, max_epochs=1, patience=3)
model.fit(X_train, y_train)
predictions = np.squeeze(model.predict(X_test))
score = model.score(X_test, y_test)
print(score)


dfb = data[['Name','Platform','Year','Genre','Publisher','Global_Sales']]
dfb = dfb.dropna().reset_index(drop=True)
df2 = dfb[['Platform','Year','Genre','Publisher','Global_Sales']]
df2['Hit'] = df2['Global_Sales']
df2.drop('Global_Sales', axis=1, inplace=True)

def hit(sales):
    if sales >= 1:
        return 1
    else:
        return 0

df2['Hit'] = df2['Hit'].apply(lambda x: hit(x))

# Logistic regression plot with sample of the data
n = ceil(0.05 * len(df2['Hit']))
fig, ax = plt.subplots(1,1, figsize=(12,5))
sns.regplot(x="Year", y="Hit", data=df2.sample(n=n),
            logistic=True, n_boot=500, y_jitter=.04, color="#75556c")

# df_copy = pd.get_dummies(df2)


