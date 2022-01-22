
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sbn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression

veri=pd.read_csv("veriler2.csv")#verileri okudum
#nümerik dönüşüm



deg=veri[["cinsiyet"]].values#cinsiyeti encode etmek için ayırdım

le=preprocessing.LabelEncoder()#2 farklı veri olduğundan encode kullandım
deg=le.fit_transform(deg)#değişkeni değiştirdim
deg=pd.DataFrame(data=deg,index=range(44),columns=["cinsiyet"])#concat kullanamk için data frame e dönüştürdüm

veri2=pd.read_csv("veriler2.csv")
deg2=veri2.iloc[:,0:1].values

ohe=preprocessing.OneHotEncoder()
deg2=ohe.fit_transform(deg2).toarray()
deg2=pd.DataFrame(data=deg2,index=range(44),columns=["fr","tr","us"])

#birleştirme
sonuc1=veri.drop("cinsiyet",axis=1).drop("ulke",axis=1)#veriden cinsiyet stununu düşürdüm
sonuc=pd.concat([deg2,sonuc1,deg],axis=1)#geri kalanıyla oluşturduğum nümerik veriyi birleştirdim

print(sonuc)

x=sonuc.drop("boy",axis=1)
y=sonuc[["boy"]]

x1,x2,y1,y2=train_test_split(x,y,test_size=0.33,random_state=0)

li=LinearRegression()
li.fit(x1,y1)

tahmin=li.predict(x2)
tahmin=pd.DataFrame(data=tahmin,index=range(15),columns=["tahmin edilen boy"])

y2=y2.reset_index()

son=pd.concat([y2,tahmin],axis=1)