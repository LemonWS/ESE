# this part is to calculate the equilibrium parameter adjustment matrix
from itertools import count
import numpy as np

from Cointegration import cointegration
from EquilibriumParameter import equilibrium_state_parameter_set


#,
def L_parameters(esps, state):
    no_part = len(esps)
    #print(no_part)
    l = np.ones(no_part)
    #print(no_part)
    for i in range(len(state)):
        l = (esps - state[i] + l)/2
        #print(l1)
    return l

 #   elif isinstance(n, (int, float)):
  #      for i in range(n):
 #           l = (state[i] - state_p[i] + l) / 2
#        return l


#print(L_parameters(n=4))


def long_run_equilibrium(esps_0,spss):
    esps = esps_0
    i = 0
    no_part = len(esps)
    l = np.ones(no_part)

    while cointegration(esps,spss):

        l = (esps - spss[i] + l) / 2

        esps = (esps_0 + l) / 2

        i += 1

    return esps

def long_run_equilibrium_l(esps_0,spss):

    #print(esps)
    i = 0
    no_part = len(esps_0)
    l = np.ones(no_part)

    while True:
        for i in range(len(spss)):
            l = (esps_0 - spss[i] + l) / 2

        esps = esps_0 - (l / 2)


        if cointegration(esps,spss) == False:
            break

    return esps
