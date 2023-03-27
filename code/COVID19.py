import math
import random
import pandas as pd
import numpy as np

from ArctanTransform import arctan_trans
from Effective import time_one
from EquilibriumIndex import equilibrium_index_DI, equilibrium_index_TED
from EquilibriumParameter import feature_distribution,equilibrium_state_parameter_set
from StateParameter import state_parameter_set
import matplotlib.pyplot as plt

from ToCsv import dataToCsv
from correlation import attribution_correlation, regression_models
import time

data_attribute_raw = pd.read_csv("covid/attributes.csv")

data_attribute = np.array(data_attribute_raw)

no_region = len(data_attribute[:, 0])
name_region = data_attribute[:, 0]

############################################
# for choosing attributes
############################################

# attribute = [2,8,11,12,14,15]
# attribute = [1,2,4,5]
attribute = [8]#, 11, 12, 14, 15, 17, 18]
# attribute = [19]
# attribute = [2]



no_attribute = len(attribute)
data_attribute = data_attribute[:, attribute]


############################################
# daily data

date_daily_raw = pd.read_csv("covid/daily.csv")

date_case = np.array(date_daily_raw[1:,:])
no_date = len(date_case[:, 0])

daily_state = list(map(list, zip(*date_case)))


################################################
######### test     #############################
################################################


# state paramater
spss = []
for i in range(no_date):
    sps = state_parameter_set(daily_state[i])
    sps = sps.astype(float)
    spss.append(sps)

# test correlates
x_c = data_attribute

y_c = spss[0] # 0 is the first day.
correlates = attribution_correlation(3, x_c, y_c) # 3 is ols

# equilibrium state
choice = 1
x=0
for i in range(no_attribute):
    a = feature_distribution(data_attribute[:, i], correlates[i])
    x += a


#esps = ((x / no_attribute + 1) + 1) / no_region
esps = equilibrium_state_parameter_set(x,no_attribute)

# equilibrium index
EI_DI = equilibrium_index_DI(spss[0],esps) # 0 is the first day. 1 is second day

EI_TED = equilibrium_index_TED(spss[0],esps)






