# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:39:15 2018

@author: shubham chopra
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

g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist, 'Age', bins=20)


train_df.hist('Age')


g=sns.FacetGrid(train_df,col='Survived',row='Pclass')
g.map(plt.hist,'Age',bins=20)

g=sns.FacetGrid(train_df,col='Embarked')
g.map(sns.pointplot,'Pclass','Survived','Sex', palette='deep')
g.add_legend()

train_df=train_df.drop(['Ticket','Cabin'],axis=1)
test_df=test_df.drop(['Ticket','Cabin'],axis=1)
combine=[train_df,test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}

for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)

train_df=train_df.drop(['Name','PassengerId'],axis=1)
test_df=test_df.drop(['Name'],axis=1)
combine=[train_df,test_df]

sex_mapping={'male':0,'female':1}
for dataset in combine:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)




train_df['Age'].fillna((train_df['Age'].mean()), inplace=True)
test_df['Age'].fillna((test_df['Age'].mean()), inplace=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] =4


for dataset in combine:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1


for dataset in combine:
    dataset['Isalone']=0
    dataset.loc[dataset['FamilySize']==1,'Isalone']=1
    
    
    
train_df=train_df.drop(['Parch','FamilySize','SibSp'],axis=1)  
test_df=test_df.drop(['Parch','FamilySize','SibSp'],axis=1)
combine=[train_df,test_df]

freq_port = train_df['Embarked'].mode()[0]


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)


train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    


X_train = train_df.drop(["Survived"], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(["PassengerId"], axis=1)
X_train.shape, Y_train.shape, X_test.shape

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

import statsmodels.api as sm
X2 = sm.add_constant(X_train)
est = sm.OLS(Y_train, X2)
est2 = est.fit()
print(est2.summary())



svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv(submission.csv, sep='\t')
submission.to_csv('./submission.csv', index=False)
