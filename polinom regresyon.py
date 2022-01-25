

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

#verleri okudum
veri=pd.read_csv("maaslar.csv")
#verileri giriş ve çıkış olarak böldük ve çizim için numpy dizisini aldık
x=veri.iloc[:,1:2]
y=veri[["maas"]]
X=x.values
Y=y.values
plt.scatter(X,Y,color="red")
#grafiğe baktığımızda artışın polinomsal artış 
#olduğunu gördüğümüzden girişinde polinomsal derecesini aldık
#(derece deneme yanılma)
poly=PolynomialFeatures(degree=5)

xpoly=poly.fit_transform(x)
#lineer regresyonu üssel artış ile eğittik
lr=LinearRegression()
lr.fit(xpoly,y)
#tahmin değerlerini aldık
tahmin=lr.predict(xpoly)

plt.scatter(X,Y,color="red")
plt.plot(X,tahmin,color="green")


print(lr.predict(poly.fit_transform([[4.3]])))