# -*- coding: utf-8 -*-

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

#veriyi okuduk
veri=pd.read_excel("Iris.xls")
#verileri x ve y ye atadık
x=veri.drop("iris",axis=1)
y=veri[["iris"]]

scaler=StandardScaler()
#verileri ayırdık
x1,x2,y1,y2=train_test_split(x,y,test_size=0.33,random_state=42)
#scele(ölçek) ettik
x1=scaler.fit_transform(x1)
x2=scaler.transform(x2)
#logistik regresyona öğretme işlemi
logr=LogisticRegression(random_state=10)
logr.fit(x1,y1)

tahminLog=logr.predict(x2)
#confusion matrix ile karşılaştırmasını yaptık
cmlog=confusion_matrix(y2,tahminLog)

#k-nn yöntemi
knn=KNeighborsClassifier(n_neighbors=6,metric="minkowski")
knn.fit(x1, y1)
tahminKnn=knn.predict(x2)
cmknn=confusion_matrix(y2,tahminKnn)
#svm yöntemi
svc=SVC(kernel="rbf")
svc.fit(x1,y1)
tahminSVC=svc.predict(x2)
cmSVC=confusion_matrix(y2,tahminSVC)

#Naive Bayes yöntemi
gnb=GaussianNB()
gnb.fit(x1,y1)
tahminGNB=gnb.predict(x2)
cmGNB=confusion_matrix(y2,tahminGNB)

#Karar Ağacı ile sınıflandırma 
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x1,y1)
tahminDTC=dtc.predict(x2)
cmDTC=confusion_matrix(y2,tahminDTC)

#Random Forrest ile tahmin
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x1,y1)
tahminRFC=rfc.predict(x2)
cmRFC=confusion_matrix(y2,tahminRFC)

print("logistic regresyon tahmin tutarlılığı :")
print(cmlog)
print("K-nn tahmin tutarlılığı :")
print(cmknn)
print("SVC tahmin tutarlılığı :")
print(cmSVC)
print("Naive Bayes tahmin tutarlılığı :")
print(cmGNB)
print("Karar ağacı tahmin tutarlılığı :")
print(cmDTC)
print("Random Forrest tahmin tutarlılığı :")
print(cmRFC)
