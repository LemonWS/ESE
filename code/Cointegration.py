from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import statsmodels.tsa.stattools as ts
import pandas as pd

from PredictTest import pvalue


def cointegration(ES,S):
    r = False
    p = 0.05
    lentgh = S.shape[0]
    ES_matrix = np.tile(vector, (lentgh, 1))

    for i in range(len(S)):
        test = ts.coint(ES[i,:],S[i,:])
        if test == None:
            test = 0
        #print(test[1])

        if test[1] > p:
            r = True
            #print(r)
            break


    return r




def weight_order(data, target, condition, n_order=5):
    worder=[]
    for i in range(n_order):
        x = np.max(data[:, target])
        y = data.loc(condition,x)
        data = np.delete[y,:]
        worder.append(y)
    return worder



