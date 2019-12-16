import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from util_helper import get_stocks_yahoo, minimize_risk, get_ret_std

update_sources = False
stocks = ['GOOG', 'NKE', 'LMT', 'CAT', 'TSLA', 'AMZN', 'TMO', 'CSIQ', 'HPE', 'MU', 'DELL', 'AMD', 'BHC', 'ROK', 'TTWO']
start = '2016-01-01'
end = '2019-10-31'
trading_days = 252
risk_free_rate = 1.8 / 100
index = 7.8 / 100

############################################################
# Import data
############################################################

if update_sources:
    # Import historical stock data from Yahoo finance
    data = get_stocks_yahoo(stocks, start, end, 'Adj Close')
    data.to_csv('out/data_master.csv')

    # Import historical betas from Yahoo finance
    betas = []
    for stock in stocks:
        beta = pd.read_html(f'https://ca.finance.yahoo.com/quote/{stock}?p={stock}')[1].iloc[1, 1]
        if stock == 'DELL':
            betas.append(float(0.74))
        else:
            if stock == 'MU':
                betas.append(float(1.8))
            else:
                betas.append(float(beta))
    pd.DataFrame(betas).to_csv('out/betas_master.csv')

else:
    # Import historical stock data from local file
    data = pd.read_csv('out/data_master.csv', index_col=0)
    print('Imported stock data...')

    betas = pd.read_csv('out/betas_master.csv', index_col=0)
    print('Imported beta data...')

############################################################
# Intermediate Calculations
############################################################

# daily historical returns for each stock (S(i) - S(i-1))/ S(i-1)) - use for calculating covariance matrix
covMatrix = data.pct_change().cov() * trading_days

# calculate the expected return on each stock according to the CAPM pricing model formula
returns = risk_free_rate + np.array(betas.iloc[:, 0]) * (index - risk_free_rate)

############################################################
# Setup Solver for initial portfolio
############################################################

# Initial Guess (equal proportion)
init_guess = np.array([1 / len(stocks)] * len(stocks))

# weights boundaries
bounds = tuple((0, 1) for asset in range(len(stocks)))

# Create a linspace number of points to calculate x
frontier_returns = np.linspace(0.06, 0.20, 50)

frontier_std = []
weights1 = []

for possible_return in frontier_returns:
    # function for finding minimum risk (standard deviation) of any given return
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std(w, returns, covMatrix)[0] - possible_return}]

    result = minimize(minimize_risk, init_guess, args=(returns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

    frontier_std.append(result['fun'])
    weights1.append(result['x'])

############################################################
# Setup Solver for risk free portfolio
############################################################
stocks.append('risk_free')

data['risk_free'] = risk_free_rate/trading_days

covMatrix = data.pct_change().cov() * trading_days

returns = np.append(returns, risk_free_rate)

init_guess = np.array([1 / len(stocks)] * len(stocks))

bounds = tuple((0, 1) for asset in range(len(stocks)))

frontier_rf_std = []
weights2 = []

for possible_return in frontier_returns:
    # function for finding minimum risk (standard deviation) of any given return
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std(w, returns, covMatrix)[0] - possible_return}]

    result = minimize(minimize_risk, init_guess, args=(returns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

    frontier_rf_std.append(result['fun'])
    weights2.append(result['x'])

print(pd.DataFrame(weights1).to_csv('out/weights1.csv'))
print(pd.DataFrame(weights2).to_csv('out/weights2.csv'))
plt.figure(figsize=(12, 8))
plt.plot(frontier_std, frontier_returns, 'b', linewidth=2, alpha=0.8)
plt.plot(frontier_rf_std, frontier_returns, 'r', linewidth=2, alpha=0.8)
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
plt.show()
