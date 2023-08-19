import math
import random
import pandas as pd
import numpy as np

from ArctanTransform import arctan_trans
from AttributionSelection import select_attribute
from Effective import time_one
from EquilibriumIndex import equilibrium_index_DI, equilibrium_index_TED
from EquilibriumParameter import feature_distribution, equilibrium_state_parameter_set
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
# daily data

date_daily_raw = pd.read_csv("covid/daily.csv")

date_case = np.array(date_daily_raw[1:,:])
no_date = len(date_case[:, 0])

daily_state = list(map(list, zip(*date_case)))

############################################
# for choosing attributes
############################################
time_record = []
#start_time = time.time()
# attribute = [2,8,11,12,14,15]
# attribute = [1,2,4,5]
#attribute = [8]#, 11, 12, 14, 15]
# attribute = [2]
num_attribution = np.shape(data_attribute[:,2:16])[1]
attribute = select_attribute(num_attribution, data_attribute[:,2:16], daily_state, 0, h=100) # h is the length of training day


no_attribute = len(attribute)
data_attribute = data_attribute[:, attribute]





################################################
######### test     #############################
################################################
start_time = time.time()

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

'''
ES= np.array(esps)
ES= ES.T
np.array(ES)
save = pd.DataFrame(ES, columns=['ES'])
save.to_csv('ES.csv', index=False, header=False)
'''

# equilibrium index
EI_DI = equilibrium_index_DI(spss[0],esps) # 0 is the first day. 1 is second day

EI_TED = equilibrium_index_TED(spss[0],esps)

'''
ES_EI_DI= np.array(EI_DI)
ES_EI_DI= ES_EI_DI.T
np.array(ES_EI_DI)
save = pd.DataFrame(ES_EI_DI, columns=['EI_DI'])
save.to_csv('EI_DI.csv', index=False, header=False)

ES_EI_TED= np.array(EI_TED)
ES_EI_TED= ES_EI_TED.T
np.array(ES_EI_TED)
save = pd.DataFrame(ES_EI_TED, columns=['EI_TED'])
save.to_csv('EI_TED.csv', index=False, header=False)
'''
end_time = time.time()  # 1657267201.6171696

time1 = end_time - start_time
time_record.append(time1)
time_record = np.array(time_record)
time_record = time_record.T

np.array(time_record)
save = pd.DataFrame(time_record, columns=['time'])
save.to_csv('time.csv', index=False, header=False)
