import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# I'm working with Ian Martin's simple variance swaps because they're centered moments 
#    (=de-trended fluctuations -> corrects for non-stationarity)
df = pd.read_csv('mfiv_smfiv_spx.csv')
df.index = df['date']
df = df[['smfiv30', 'smfiv91', 'smfiv182', 'smfiv273', 'smfiv365']] 

# 1. variance swaps are annualized -> de-annualize them
#      (= total implied variance)
tau = np.array([30, 91, 182, 273, 365]) / 365
TIV = np.matmul(df, np.diag(tau))

# 2. take logs
x = np.log(tau)
x = sm.add_constant(x)
y = np.log(TIV)

# 3. estimate the slope of the log implied var. term-structure at each day
#     = implied Hurst exponent -> a measure of auto-correlation
#       Hurst can be transformed to Pearson's auto-corr, 
H = [] # Hurst
ac = [] # Pearson's auto-corr. (lag = 1)
k = 1
for date, row in y.iterrows():
  mdl = sm.OLS(row, x).fit()
  Ht = 0.5 * mdl.params.iloc[1]
  act = 0.5*(np.abs(k+1)**(2*Ht) - 2*np.abs(k)**(2*Ht) + np.abs(k-1)**(2*Ht))
  H.append(Ht)
  ac.append(act)

plt.plot(ac)   # ranges in [-1, 1]
plt.plot(H)    # ranges in [0, 1]
plt.show()
