# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 01:05:59 2020

@author: polat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
veri=pd.read_csv("maaslar.csv")
x=veri[["Egitim Seviyesi"]]
y=veri[["maas"]]

rfr=RandomForestRegressor(n_estimators=100,random_state=10 )

rfr.fit(x,y)

tahmin=rfr.predict(x)
