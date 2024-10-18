import pandas as pd
import numpy as np
from AttributesSelection import attribution_select_TED, attribution_select_Dis, transpose_list
from EquilibriumParameter import feature_distribution, feature_distribution_up_0, equilibrium_state_parameter_set
from LongRunTraining import L_parameters, long_run_equilibrium, long_run_equilibrium_l
from PredictiorForPoint import ESE_predictor_system_ar
from StateParameter import state_parameter_set
from correlation import attribution_correlation, regression_models, attribution_correlate_coe

data_raw_attribute = pd.read_csv("demo/attribute/try1.csv")

data_raw_attribute = np.array(data_raw_attribute)

no_part = 504  # the number of all part
name_part = data_raw_attribute[:, 0]

###########################################
# setting for time period
###########################################

start = -200  # ceg ipo only 60 days
end = -100
no_date = end - start

######## load daily data ################

date = pd.read_csv("demo/A.csv")
date = np.array(date)
date = date[start:end, 0].astype(str)

############################################
# for choosing attributes
############################################

attribute_list = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}#, 12, 13, 14, 15, 16, 17, 18, 19}



no_name = range(len(name_part)) # number of part names

# number_name = range(len(name_part)) 
daily_data_raw = []          #### !!! revise raw_data -> a matrix of raw records, each column represents a time unit, e.g. a day or an hour, each row represnts data of a part/system at different time units ###

daily_data_total_raw = []    #### raw_data_sum -> 1D array, each element is the sum of all parts at a particular time point


##### load daily data raw and attribute data #####
for i in no_name:
    share = pd.read_csv("demo/" + name_part[i] + ".csv")   #read the daily data
    part = np.array(share)
    data = part[start:end, 5].astype(float) ### 5 is hard code  target data column
    daily_data_raw.append(data)

###  Transpose the data matrix so each row is systems/parts at a particular time point, 
###  while each column is the time series of a system/part 
daily_state = list(map(list, zip(*daily_data_raw)))

### Calculate the sum of all parts, e.g. sum of one row
for i in range(len(daily_state)):
    total = sum(daily_state[i])
    daily_data_total_raw.append(total)

### spss is a collection of sps, the state parameter set for one time point
spss = []
for i in range(len(daily_data_raw[0])):  # test_start+h_start-1 is t_0
    sps = state_parameter_set(daily_state[i])
    sps = sps.astype(float)
    spss.append(sps)

### attribute selection based on Dis
attribute_set = attribution_select_Dis(data_raw_attribute, spss, attribute_list)
no_attribute = len(attribute_set)

### data of selected attributes
at = data_raw_attribute[:,attribute_set]

### Calculate the correlation between the state parameter set and the last data point which is the prediction target 
### 3 is the switch value for OLS
correlates = attribution_correlate_coe(3, at, spss[-1])

### sum of the absoluton values of correlations
sum_cor = sum(abs(correlates))  ### sum_correlation

### generate feature values for each attribute under all systems/parts
x = 0
for i in range(no_attribute):
    a = feature_distribution_up_0(at[:, i], correlates[i])
    x += a

### equilibrium sps, initial value
esps_0 = equilibrium_state_parameter_set(sum_cor, no_attribute, no_part, x)

### long run equilibirum training
esps = long_run_equilibrium_l(esps_0,spss)

### prediction based on the trained esps
p = ESE_predictor_system_ar(daily_data_total_raw,esps)  ### default prediction distance = 1

### p = daily_data_total_raw[-1]*esps ### prediction with just esps based on historical data

