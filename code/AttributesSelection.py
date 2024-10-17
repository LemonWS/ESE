# the function for selecting the attribution
import math

import numpy as np

from EquilibriumIndex import equilibrium_index_TED, equilibrium_index_DI
from EquilibriumParameter import feature_distribution_up_0, equilibrium_state_parameter_set
from correlation import attribution_correlate_coe

def transpose_list(matrix):
    return [list(row) for row in zip(*matrix)]
def attribution_select_TED(attributes, state, list):
    EI = 1
    no_part = len(attributes)
    #print(no_part)
    alt = subsets(list)
    #print(alt)
    num = len(alt)
    #print(num)
    for i in range(1,num):
        at_column = attributes[:,alt[i]]
        no_attribute = len(alt[i])
        #print(no_attribute)
        correlates = attribution_correlate_coe(3, at_column, state[-1])
        sum_cor = sum(abs(correlates))
        x = 0
        for i in range(no_attribute):
            a = feature_distribution_up_0(at_column[:, i], correlates[i])
            #print(sum(a))
            x += a
            #print(x)
        esps = equilibrium_state_parameter_set(sum_cor, no_attribute, no_part, x)
        #print(sum(esps))

        EI_TED = equilibrium_index_TED(state[-1], esps)
        if EI_TED < EI:
            EI = EI_TED
            attribute_set = alt[i]
        else:
            EI = EI

    return attribute_set


def attribution_select_Dis(attributes, state, list):
    EI = 1
    no_part = len(attributes)
    alt = subsets(list)
    #print(alt)
    num = len(alt)
    #print(num)
    for i in range(1,num):
        at_column = attributes[:,alt[i]]
        no_attribute = len(alt[i])
        #print(no_attribute)
        correlates = attribution_correlate_coe(3, at_column, state[-1])
        sum_cor = sum(abs(correlates))
        x = 0
        for i in range(no_attribute):
            a = feature_distribution_up_0(at_column[:, i], correlates[i])
            #print(sum(a))
            x += a
            #print(x)
        esps = equilibrium_state_parameter_set(sum_cor, no_attribute, no_part, x)
        #print(sum(esps))
        EI_DiS = equilibrium_index_DI(state[-1], esps)
        #print(EI_DiS)
        if EI_DiS < EI:
            EI = EI_DiS
            attributeset = alt[i]
        else:
            EI = EI
    return attributeset



def subsets(nums):
    result = [[]]
    for num in nums:
        for element in result[:]:
            x = element[:]
            x.append(num)
            result.append(x)
    return result