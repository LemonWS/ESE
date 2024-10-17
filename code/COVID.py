import pandas as pd
import numpy as np
from AttributesSelection import attribution_select_TED, attribution_select_Dis, transpose_list
from EquilibriumParameter import feature_distribution, feature_distribution_up_0, equilibrium_state_parameter_set
from LongRunTraining import L_parameters, long_run_equilibrium, long_run_equilibrium_l
from PredictiorForPoint import ESE_predictor_system_ar
from StateParameter import state_parameter_set
from correlation import attribution_correlation, regression_models, attribution_correlate_coe
import time


no_part = 20#,79,320  # the number of all part

data_raw_attribute = pd.read_csv("COVID/attribute/attribute_20.csv")
#data_raw_attribute = pd.read_csv("COVID/attribute/attribute_79.csv")
#data_raw_attribute = pd.read_csv("COVID/attribute/attribute_320.csv")

data_raw_attribute = np.array(data_raw_attribute)

name_part = data_raw_attribute[:, 0]

###########################################
# setting for time period
###########################################

start = -200  # ceg ipo only 60 days
end = -100
no_date = end - start

######## lord daily data ################

date = pd.read_csv("COVID/Alpine.csv")
date = np.array(date)
date = date[start:end, 0].astype(str)

############################################
# for choosing attributes
############################################

attribute_list = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}#, 12, 13, 14, 15, 16, 17, 18, 19}
#attribute_list = {2} only for 320


no_name = range(len(name_part))

number_name = range(len(name_part))

daily_data_raw = []

daily_data_total_raw = []

time_record = []

##### lord daily and attribute #####

for i in no_name:
    share = pd.read_csv("COVID/" + name_part[i] + ".csv")  #read the daily data
    part = np.array(share)
    data = part[start:end, 5].astype(float)
    daily_data_raw.append(data)

daily_state = list(map(list, zip(*daily_data_raw)))

for i in range(len(daily_state)):
    total = sum(daily_state[i])
    daily_data_total_raw.append(total)

spss = []
for i in range(len(daily_data_raw[0])):  # test_start+h_start-1 is t_0
    sps = state_parameter_set(daily_state[i])
    sps = sps.astype(float)
    spss.append(sps)


test_model_choice = 1  # if 1, no loop, 2 loop and only for select_stock_item = 2
select_stock_item = 1  # 1 for all, 2 for select order, 3 for select random， 4 for select specific stocks
loop_time_set = 100  # if 100, it means loop 100 times

if test_model_choice == 1:
    loop_time = 1
else:
    select_stock_item = 2
    loop_time = loop_time_set

select_order = 0  # if 10,it means first 10 stocks in order
time_record = []

for i in range(loop_time):
    select_order += 5

    start_time = time.time()

    attribute_set = attribution_select_Dis(data_raw_attribute, spss, attribute_list)
    #attribute_list = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}  # , 12, 13, 14, 15, 16, 17, 18, 19} # test time only
    no_attribute = len(attribute_set)

    at = data_raw_attribute[:, attribute_set]

    correlates = attribution_correlate_coe(3, at, spss[-1])

    sum_cor = sum(abs(correlates))

    x = 0
    for i in range(no_attribute):
        a = feature_distribution_up_0(at[:, i], correlates[i])
        x += a

    esps_0 = equilibrium_state_parameter_set(sum_cor, no_attribute, no_part, x)

    esps = long_run_equilibrium_l(esps_0, spss)

    p = ESE_predictor_system_ar(daily_data_total_raw, esps)

    end_time = time.time()  # 1657267201.6171696

    time1 = end_time - start_time
    time_record.append(time1)



time_record = np.array(time_record)
time_record = time_record.T



np.array(time_record)
save = pd.DataFrame(time_record, columns=['time'])
save.to_csv('time.csv', index=False, header=False)
print('finish')