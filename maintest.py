import pandas as pd
import datetime
from scipy.optimize import _minimize

from util.findFiles import findfiles

print(datetime.datetime.now())

path = 'data/KL.TO.csv'
targetDir = 'data'
data = []

# locate and import all data files in the 'data' directory
# store imported data in the data list (type DataFrame)
for file in findfiles(targetDir):
    data.append(pd.read_csv(file, encoding='latin1'))

data_clean = []
# convert Date to datetime format
for d in data:
    d.Date = pd.to_datetime(d.Date, infer_datetime_format=True)
    d.set_index('Date', inplace=True, drop=True)
    data_clean.append(d)
    # print(d.head())

# data_master = data_clean[0].join(data_clean[1], on='Date', rsuffix='_1')
data_master = data_clean[0].join(data_clean[1], lsuffix='_0', rsuffix='_1')
data_master.to_csv('out/data_master.csv')
data_returns = pd.DataFrame()
# calculate daily returns
for i in range(len(data_clean)):
    data_returns['daily_return_' + str(i)] = data_master['Adj Close_' + str(i)].pct_change()

print(data_returns.head())
print(data_returns.info())

# calculate covariance between daily returns, multiply by number of trading days per year
cov = data_returns.cov() * 252
print(cov)

frontier_std = []

for possible_return in frontier_returns:
    # function for finding minimum risk (standard deviation) of any given return
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std_sr(returns, w, covarianceMatrix)[0] - possible_return})

    result = minimize(minimize_risk, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    frontier_std.append(result['fun'])

plt.figure(figsize=(12, 8))
plt.scatter(std_arr, ret_arr, c=sharpe_arr, cmap='plasma_r')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation')
plt.ylabel('Return')