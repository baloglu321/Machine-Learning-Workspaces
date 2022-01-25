
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression

veri=pd.read_csv("satislar.csv")
print(veri)

sbn.scatterplot(data=veri,x="Aylar",y="Satislar")

x=veri[["Aylar"]]
y=veri[["Satislar"]]

x1,x2,y1,y2=train_test_split(x,y,test_size=0.3,random_state=0)
'''
scaler=StandardScaler()

x1=scaler.fit_transform(X1)
x2=scaler.fit_transform(X2)
'''
lr=LinearRegression()

lr.fit(x1,y1)

tahmin=lr.predict(x2)

x1=x1.sort_index()
y1=y1.sort_index()

plt.plot(x1,y1)
plt.plot(x2,tahmin)
