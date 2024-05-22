from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd

def cointegration(ES_list,S_list):
    for i in range(len(ES_list)):
        johansen_test = coint_johansen(ES_list[i],S_list[i],det_order=0,k_ar_diff=1)
        if johansen_test.lr1[0] > johansen_test.cvt[0, 1]:
            return True
        else:
            return False