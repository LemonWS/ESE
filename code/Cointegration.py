from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import statsmodels.tsa.stattools as ts
import pandas as pd

def cointegration(ES,S):
    r = False
    for i in range(len(S)):
        test = ts.coint(ES,S[i])
        if test[1] > 0.05:
            r = True
            print(r)

    return r




def weight_order(data, target, condition, n_order=5):
    worder=[]
    for i in range(n_order):
        x = np.max(data[:, target])
        y = data.loc(condition,x)
        data = np.delete[y,:]
        worder.append(y)
    return worder



