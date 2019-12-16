import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from util_helper import import_data_and_betas, minimize_risk, get_ret_std

stocks = ['GOOG', 'NKE', 'LMT', 'CAT', 'TSLA', 'AMZN', 'TMO', 'CSIQ', 'HPE', 'MU', 'DELL', 'AMD', 'BHC', 'ROK', 'TTWO']
start = '2017-01-01'
end = '2019-11-30'
trading_days = 252
risk_free_rate = 1.8 / 100
index = 7.8 / 100

############################################################
# Import data
############################################################
data , betas = import_data_and_betas(stocks, start, end, update_sources = False)


############################################################
# Intermediate Calculations
############################################################

# daily historical returns for each stock (S(i) - S(i-1))/ S(i-1)) - use for calculating covariance matrix
covMatrix = data.pct_change().cov() * trading_days

# calculate the expected return on each stock according to the CAPM pricing model formula
returns = risk_free_rate + betas * (index - risk_free_rate)


############################################################
# Create some random portfolios
############################################################
num_portfolios = 2000

all_weights = np.zeros((num_portfolios,len(stocks)))
ret_arr = np.zeros(num_portfolios)
std_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)


for n in range(num_portfolios):

    # Create Random Weights
    weights = np.array(np.random.random(len(stocks)))

    # Rebalance Weights (sum equal to 1)
    weights = weights / np.sum(weights)
    
    all_weights[n,:] = weights

    # Expected Return
    ret_arr[n] = np.sum(returns * weights)

    # Expected Standard deviation
    std_arr[n] = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))

    # Sharpe Ratio
    sharpe_arr[n] = ret_arr[n]/std_arr[n]


############################################################
# Setup Solver for initial portfolio
############################################################
    
# Initial Guess (equal proportion)
init_guess = np.array([1 / len(stocks)] * len(stocks))

# weights boundaries
bounds = tuple((0, 1) for asset in range(len(stocks)))

# Create a linspace number of points to calculate x
frontier_returns = np.linspace(min(ret_arr),max(ret_arr),20) 

frontier_std = []
weights_frintier = []

for possible_return in frontier_returns:
    # function for finding minimum risk (standard deviation) of any given return
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std(w, returns, covMatrix)[0] - possible_return}]

    result = minimize(minimize_risk, init_guess, args=(returns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

    frontier_std.append(result['fun'])
    weights_frintier.append(result['x'])


plt.figure(figsize=(12,8))
plt.scatter(std_arr,ret_arr,c=sharpe_arr,cmap='plasma_r' )
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
# Add frontier line
plt.plot(frontier_std,frontier_returns,'b--',linewidth=3)
plt.show()


############################################################
# Setup Solver for risk free portfolio
############################################################
stocks.append('risk_free')

data['risk_free'] = risk_free_rate/trading_days

covMatrix = data.pct_change().cov() * trading_days

returns = np.append(returns, risk_free_rate)

init_guess = np.array([1 / len(stocks)] * len(stocks))

bounds = tuple((0, 1) for asset in range(len(stocks)))

frontier_std_with_RF = []
weights_frintier_with_RF = []

for possible_return in frontier_returns:
    # function for finding minimum risk (standard deviation) of any given return
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std(w, returns, covMatrix)[0] - possible_return}]

    result = minimize(minimize_risk, init_guess, args=(returns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

    frontier_std_with_RF.append(result['fun'])
    weights_frintier_with_RF.append(result['x'])

print(pd.DataFrame(weights_frintier).to_csv('out/weights1.csv'))
print(pd.DataFrame(weights_frintier_with_RF).to_csv('out/weights2.csv'))


plt.figure(figsize=(8,6))
plt.plot(frontier_std,frontier_returns,'b--',linewidth=3)
plt.plot(frontier_std_with_RF,frontier_returns,'g-',linewidth=2)
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
plt.savefig('out/efficient_frontier_plot')
plt.show()


results = pd.DataFrame({'Return' :frontier_returns, 'Minimum_Risk':frontier_std_with_RF, 'w': weights_frintier_with_RF} )
w = pd.DataFrame(results['w'].values.tolist(), columns=stocks)
results = results.join(w).drop('w' , axis = 1).round(3)

results.to_csv('out/results.csv')