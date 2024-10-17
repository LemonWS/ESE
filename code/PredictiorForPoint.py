# the predictor for ESE, as time series
from statsmodels.tsa.ar_model import AutoReg



def ESE_predictor_part(part, state, equilibrium):

    diff = (state-equilibrium) / state
    p = part * (1 + diff)

    return p

def ESE_predictor_system(w, equilibrium):
    ps = []
    for i in range(len(equilibrium)):
        p = w * equilibrium[i]
        ps.append(p)
    return ps

def ESE_predictor_system_ar(w, equilibrium):
    ps = []
    mod = AutoReg(w,lags=1).fit()
    fcast = mod.forecast()
    #print(fcast)
    for i in range(len(equilibrium)):
        p = fcast * equilibrium[i]
        ps.append(p)
    return ps
#def ESE_predictor_system_ar(w, equilibrium,h):
#    model = sm.tsa.AR(w)
 #   result = model.fit(maxlag=1)
 #   forecast = result.predict(start=len(w), end=len(w) + h)
  #  for i in range(len(equilibrium)):
   #     p = mod.params[1] * w_traning[:-1] * equilibrium[i]
   #     ps.append(p)
 #   return forecast