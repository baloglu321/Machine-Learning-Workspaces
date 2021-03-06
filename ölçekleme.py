
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

veri2=pd.read_csv("veriler.csv")
deg2=veri2.iloc[:,0:1].values

ohe=preprocessing.OneHotEncoder()
deg2=ohe.fit_transform(deg2).toarray()

deg2=pd.DataFrame(data=deg2,index=range(22),columns=["fr","tr","us"])
sonuc1=pd.DataFrame(data=veri2.drop("ulke",axis=1))

sonuc=pd.concat([deg2,sonuc1],axis=1)


x=sonuc.drop("cinsiyet",axis=1)
y=sonuc[["cinsiyet"]]

x1,x2,y1,y2=train_test_split(x,y,test_size=0.3,random_state=15)
#scaler standart yerine MinMaxScaler da kullanılabilir- verileri 0-1 aralığına ölçekler
scaler=StandardScaler()

x1=scaler.fit_transform(x1)
x2=scaler.fit_transform(x2)