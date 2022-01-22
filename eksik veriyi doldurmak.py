# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("eksikveriler.csv")

dtf=pd.DataFrame(veriler)
dtfyasort=dtf[["yas"]].mean()
yas=dtf[["yas"]].fillna(dtfkiloort)
dtf[["yas"]]=yas
print(dtf)


