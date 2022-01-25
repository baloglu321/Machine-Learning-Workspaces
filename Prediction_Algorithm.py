
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

veri=pd.read_csv("maaslar2.csv")
xdeg=veri[["UnvanSeviyesi"]].sort_values("UnvanSeviyesi")
#encode

ünvan=veri[["unvan"]]

ohe=preprocessing.OneHotEncoder()
deg=ohe.fit_transform(ünvan).toarray()

deg=pd.DataFrame(data=deg,index=range(30),columns=["C-level","CEO" ,"Cayci", "Direktor" , "Mudur", "Proje Yoneticisi" ,"Sef", "Sekreter", "Uzman","Uzman Yardimcisi" ])

sonuc1=veri.drop("unvan",axis=1)

sonuc=pd.concat([deg,sonuc1],axis=1)

x=sonuc.drop("maas",axis=1).drop("Calisan ID",axis=1)
y=sonuc[["maas"]]


Y=y.values


a=np.append(arr=np.ones((30,1)).astype(int),values=sonuc.iloc[:,:-1],axis=1)
xL=sonuc.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13]].values
xL=np.array(xL,dtype=float)
model=sm.OLS(y,xL).fit()
print(model.summary())





#çoklu regresyon

lr=LinearRegression()
lr.fit(x,y)

tahminMLR=lr.predict(x)

#polinom regresyon
xreg=sonuc[["UnvanSeviyesi"]]
poly=PolynomialFeatures(degree=8)

xreg=poly.fit_transform(xreg)


polylr=LinearRegression()
polylr.fit(xreg, y)

tahminpoly=polylr.predict(xreg)
#svr

scaler=StandardScaler()
scalex=scaler.fit_transform(x)
scaley=scaler.fit_transform(y)

svrR=SVR(kernel="rbf")
svrR.fit(scalex,scaley)
tahminsvr=svrR.predict(scalex)

#decision tree(karar ağacı)
dtr=DecisionTreeRegressor(random_state=15)
dtr.fit(x,y)

tahmindtr=dtr.predict(x)

#random forrest

rfr=RandomForestRegressor(n_estimators=200,random_state=15)
rfr.fit(x,y)
tahminrfr=rfr.predict(x)

#değerlendirmeler
print(f"çoklu lineer tahmin %: {r2_score(y,tahminMLR)}" )
print(f"polinom tahmin %: {r2_score(y,tahminpoly)}" )
print(f"svr tahmin %: {r2_score(scaley,tahminsvr)}" )
print(f"karar ağacı tahmin %: {r2_score(y,tahmindtr)}" )
print(f"random forrest tahmin %: {r2_score(y,tahminrfr)}" )

