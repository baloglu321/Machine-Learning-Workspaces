# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:34:06 2020

@author: polat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import statsmodels.api as sm
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import r2_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import re
import nltk

import keras
from keras.models import Sequential
from keras.layers import Dense


veri=pd.read_csv("Churn_Modelling.csv")

deg=veri[["Geography"]]
deg1=veri[["Gender"]]

le=preprocessing.LabelEncoder()
Gender=le.fit_transform(deg1)
Gender=pd.DataFrame(data=Gender,columns=["Gender"])

ohe=preprocessing.OneHotEncoder()
sonuc=ohe.fit_transform(deg).toarray()

CreditScore=veri[["CreditScore"]]
Geography=pd.DataFrame(data=sonuc,index=range(0,10000),columns=["France","Germany","Spain"])
deg2=veri.iloc[:,6:13]

x=pd.concat([Geography,Gender,deg2,CreditScore],axis=1)
y=veri[["Exited"]]

x1,x2,y1,y2=train_test_split(x,y,test_size=0.33,random_state=12)

scaler=StandardScaler()

scalex1=scaler.fit_transform(x1)

scalex2=scaler.fit_transform(x2)

model=Sequential()

model.add(Dense(6, activation="relu",input_dim=12))

model.add(Dense(6, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(scalex1,y1,epochs=50)

tahmin=model.predict(scalex2)


cm=confusion_matrix(y2,tahmin>0.5)

print(cm)
