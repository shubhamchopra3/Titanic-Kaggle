# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:31:11 2018

@author: srchopra
"""

#importing data analysis libraries 
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import random as rnd
import seaborn as sns


#importing machine learning libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Reading data
train_df= pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
combine=[train_df,test_df]

#getting feature names 
print(list(train_df))

#preview the data and see which features are categoreical and to see which feature contains empty values
train_df.head()
train_df.tail()
train_df.info()
test_df.info()

#to count the number of people survived, people with no children,parents, relatives
train_df['Survived'].value_counts()
train_df['SibSp'].value_counts()
train_df['Parch'].value_counts()


"""train_df.describe()
train_df.describe(include=['O'])"""  # this describes all object columns (categorical columns)
train_df.describe(include='all')


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df,col='Pclass')
g.map(plt.hist, 'Age', bins=20)


train_df.hist('Age')











