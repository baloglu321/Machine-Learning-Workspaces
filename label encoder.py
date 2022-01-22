# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:54:03 2020

@author: polat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#verileri encode etmek:
    
veri=pd.read_csv("veriler.csv")
deg=veri.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
deg[:,0]=le.fit_transform(veri.iloc[:,0])
veri.iloc[:,0]=deg
print(veri)

#verileri one hot encode etmek (verileri adreslemek)
veri2=pd.read_csv("veriler.csv")
deg2=veri2.iloc[:,0:1].values

ohe=preprocessing.OneHotEncoder()
deg2=ohe.fit_transform(deg2).toarray()


#yanyana yapıştır

deg2=pd.DataFrame(data=deg2,index=range(22),columns=["fr","tr","us"])
sonuc1=pd.DataFrame(data=veri.drop("ulke",axis=1))

sonuc=pd.concat([deg2,sonuc1],axis=1)


print(sonuc)

