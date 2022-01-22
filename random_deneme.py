# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:45:39 2020

@author: polat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veri=pd.read_csv("reklamverileri.csv")


N=10000
d=10
t=0
secim=[]
import random

for i in range(0,N):
    ad=random.randrange(0,d)
    secim.append(ad)
    p=veri.values[i,ad]
    t=t+p
    