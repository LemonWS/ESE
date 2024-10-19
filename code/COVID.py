import pandas as pd
import numpy as np
from AttributesSelection import attribution_select_Dis, attribution_select
from EquilibriumParameter import feature_distribution_up_0, equilibrium_state_parameter_set
from LongRunTraining import long_run_equilibrium_l
from PredictiorForPoint import ESE_predictor_system_ar
from StateParameter import state_parameter_set
from Correlation import attribution_correlate_coe
import time


number_part = 20#,79,320  # the number of all part

multi_system = "region_20/" # "region_79/" "region_320/" # the type of multi_system

raw_data_attribute = pd.read_csv("COVID/attribute/attribute_20.csv") # load the attribute for 20 region
#raw_data_attribute = pd.read_csv("COVID/attribute/attribute_79.csv") # load the attribute for 20 region
#raw_data_attribute = pd.read_csv("COVID/attribute/attribute_320.csv") # load the attribute for 20 region

raw_data_attribute = np.array(raw_data_attribute)

name_part = raw_data_attribute[:, 0]

###########################################
# setting for time period
###########################################

start = -200  # ceg ipo only 60 days
end = -100
number_time_unit = end - start

######## lord daily data ################

#date = pd.read_csv("COVID/Alpine.csv")
#date = np.array(date)
#date = date[start:end, 0].astype(str)

############################################
# for choosing attributes
############################################

attribute_list = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}#, 12, 13, 14, 15, 16, 17, 18, 19}
#attribute_list = {2} only for 320


#number_part = range(len(name_part))

#number_name = range(len(name_part))

raw_data = []

raw_data_sum = []

time_record = []

##### lord daily and attribute #####
target_data = 4# select the target data



for i in range(number_part):
    share = pd.read_csv("COVID/" + multi_system + name_part[i] + ".csv")  #read the daily data
    part = np.array(share)
    data = part[start:end, target_data].astype(float)
    raw_data.append(data)

daily_state = list(map(list, zip(*raw_data)))

for i in range(len(daily_state)):
    total = sum(daily_state[i])
    raw_data_sum.append(total)

spss = []
for i in range(len(raw_data[0])):  # test_start+h_start-1 is t_0
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


    method_choose = 1  # 1 is the EI method based on Euclidean distance, 2 is based on difference

    attribute_set = attribution_select(method_choose,raw_data_attribute, spss, attribute_list)
    #attribute_list = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}  # , 12, 13, 14, 15, 16, 17, 18, 19} # test time only
    number_attribute = len(attribute_set)

    selected_attribute_set = raw_data_attribute[:, attribute_set]


    model = 3  ### 3 is the switch value for OLS
    correlates = attribution_correlate_coe(model, selected_attribute_set, spss[-1])

    sum_cor = sum(abs(correlates))

    x = 0
    for i in range(number_attribute):
        a = feature_distribution_up_0(selected_attribute_set[:, i], correlates[i])
        x += a

    esps_0 = equilibrium_state_parameter_set(sum_cor, number_attribute, number_part, x)

    esps = long_run_equilibrium_l(esps_0, spss)

    p = ESE_predictor_system_ar(raw_data_sum, esps)

    end_time = time.time()  # 1657267201.6171696

    time1 = end_time - start_time
    time_record.append(time1)



time_record = np.array(time_record)
time_record = time_record.T



np.array(time_record)
save = pd.DataFrame(time_record, columns=['time'])
save.to_csv('time.csv', index=False, header=False)
print('finish')