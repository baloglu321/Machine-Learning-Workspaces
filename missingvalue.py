

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

veriler=pd.read_csv("eksikveriler.csv")

dtf=pd.DataFrame(veriler)

imputer= SimpleImputer(missing_values=(np.nan), strategy="mean")

deg=dtf.iloc[:,1:4].values
imputer=imputer.fit(deg)
deg=imputer.transform(deg)
dtf.iloc[:,1:4]=deg
print(dtf)

