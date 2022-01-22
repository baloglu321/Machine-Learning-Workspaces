# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:40:19 2020

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
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer



veri=pd.read_csv("Restaurant_Reviews.csv")
point=[]
for i in range (0,1000):
    for j in range(0,6):
        a=veri.iloc[i,j]
        if a=="1"or a=="0"or a==0 or a==1:
            if not a==int:
                a=int(a)
                point.append(a)     
                
veri2=pd.read_csv("restn.csv")   
ps=PorterStemmer() 
derlem=[]   
for m in range (0,1000):      
    yorum=re.sub('[^a-zA-Z]',' ',veri2.iloc[m,0])
    yorum=yorum.lower().split()


    yorumlar=[ps.stem(k) for k in yorum if not k in set(stopwords.words('english'))]
    
    yorumlar=' '.join(yorumlar)
        
    derlem.append(yorumlar)    


cv=CountVectorizer(max_features=2000)

x=cv.fit_transform(derlem).toarray()
y=np.array(point)

x1,x2,y1,y2=train_test_split(x,y,test_size=0.2,random_state=10)

nb=GaussianNB()
nb.fit(x1,y1)

tahmin=nb.predict(x2)
cm=confusion_matrix(y2, tahmin)
print(cm)