import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

<<<<<<< Updated upstream

# get adjusted closing prices of selected stocks with yahoo
stocks = ['GOOG', 'NKE', 'LMT', 'CAT' , 'TSLA', 'AMZN', 'TMO' , 'CSIQ' , 'HPE', 'AMD' , 'BHC', 'ROK' , 'TTWO', 'DELL', 'MU']

source = 'yahoo'
start = '2017-01-01'
end = '2019-11-30'


#Retrieve stock price data
data = pd.DataFrame()
for symbol in stocks:
        data[symbol] = pdr.get_data_yahoo(symbol,  start=start, end=end)['Adj Close']
        
#data.to_csv('stocks.csv')

#daily historical returns for each stock (S(i) - S(i-1))/ S(i-1)) - use for calculating covariance matrix 
daily_historical_returns = data.pct_change()

#Calculate Returns
#according to Bloomberg’s estimate
#according to yahoo Finance
# betas = []
# for stock in stocks:
    
#     beta = pd.read_html(f'https://ca.finance.yahoo.com/quote/{stock}?p={stock}')[1].iloc[1,1]
#     betas.append(float(beta))

betas = [1.02, 0.83, 0.95, 1.51, 0.58, 1.52, 1.12, 2.04, 1.53, 3.08, 0.89, 1.44, 0.69, 0.74, 1.8]

#risk free interest rate is 1.8% & expected return on the index is 7.8%
risk_free_rate = 1.8 /100
index = 7.8 /100

#calculate the expected return on each stock according to the CAPM pricing model formula
returns = risk_free_rate + np.array(betas) * ( index - risk_free_rate)

#Creating some random portfolios
=======
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
data , betas = import_data_and_betas(update_sources = False)


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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream


#Takes in weights, returns array or return,standard deviation, sharpe ratio 
def get_ret_std_sr(weights):

    weights = np.array(weights) 
    port_return = np.sum(returns * weights)
    port_std = np.sqrt(np.dot(weights.T, np.dot(daily_historical_returns.cov() * 252, weights)))
    port_sr = port_return/port_std
    
    return np.array([port_return,port_std,port_sr])

#objective
def minimize_risk(weights):
    return  get_ret_std_sr(weights)[1]

# Initial Guess (equal porportion)
init_guess = [1/len(stocks)] *len(stocks)
=======


############################################################
# Setup Solver for initial portfolio
############################################################
    
# Initial Guess (equal proportion)
init_guess = np.array([1 / len(stocks)] * len(stocks))
>>>>>>> Stashed changes

# weights boundaries
bounds = tuple((0,1) for asset in range(len(stocks)))

# Create a linspace number of points to calculate x
frontier_returns = np.linspace(min(ret_arr),max(ret_arr),20) 

frontier_std = []
weights_frintier = []

for possible_return in frontier_returns:
    
    # function for finding minimum risk (standard deviation) of any given return
<<<<<<< Updated upstream
    constraints = ({'type':'eq','fun': lambda w: np.sum(w) - 1},
            {'type':'eq','fun': lambda w: get_ret_std_sr(w)[0] - possible_return})
    
    result = minimize(minimize_risk,init_guess,method='SLSQP',bounds=bounds,constraints=constraints)
    
=======
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'eq', 'fun': lambda w: get_ret_std(w, returns, covMatrix)[0] - possible_return}]

    result = minimize(minimize_risk, init_guess, args=(returns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream

#Portfolio with risk free asset
stocks.append('Risk_Free')

data['Risk_Free'] = risk_free_rate/252
=======

############################################################
# Setup Solver for risk free portfolio
############################################################
stocks.append('risk_free')
>>>>>>> Stashed changes

data['risk_free'] = risk_free_rate/trading_days

covMatrix = data.pct_change().cov() * trading_days

returns = np.append(returns , risk_free_rate)

<<<<<<< Updated upstream
# Initial Guess (equal porportion)
init_guess = [1/len(stocks)] *len(stocks)

# weights boundaries
bounds = tuple((0,1) for asset in range(len(stocks)))

# Create a linspace number of points to calculate x
frontier_returns = np.linspace(min(ret_arr),max(ret_arr),20) 

frontier_std_with_RF = []
weights_frintier = []
=======
init_guess = np.array([1 / len(stocks)] * len(stocks))

bounds = tuple((0, 1) for asset in range(len(stocks)))

frontier_std_with_RF = []
weights_frintier_with_RF = []
>>>>>>> Stashed changes

for possible_return in frontier_returns:
    
    # function for finding minimum risk (standard deviation) of any given return
<<<<<<< Updated upstream
    constraints = ({'type':'eq','fun': lambda w: np.sum(w) - 1},
            {'type':'eq','fun': lambda w: get_ret_std_sr(w)[0] - possible_return})
    
    result = minimize(minimize_risk,init_guess,method='SLSQP',bounds=bounds,constraints=constraints)
    
    frontier_std_with_RF.append(result['fun'])
    weights_frintier.append(result['x'])

plt.figure(figsize=(12,8))

plt.scatter(std_arr,ret_arr,c=sharpe_arr,cmap='plasma_r'  , alpha = 0.6)
plt.colorbar(label='Sharpe Ratio')
=======
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
>>>>>>> Stashed changes
plt.xlabel('Standard Deviation')
plt.ylabel('Return')
plt.savefig('out/efficient_frontier_plot')
plt.show()

<<<<<<< Updated upstream
# Add frontier line
plt.plot(frontier_std,frontier_returns,'b-',linewidth=2)
plt.plot(frontier_std_with_RF,frontier_returns,'g--',linewidth=4)
plt.show()


plt.figure(figsize=(8,6))

plt.xlabel('Standard Deviation')
plt.ylabel('Return')

# Add frontier line
plt.plot(frontier_std,frontier_returns,'b-',linewidth=2)
plt.plot(frontier_std_with_RF,frontier_returns,'g--',linewidth=4)
plt.show()

results = pd.DataFrame({'return' :frontier_returns, 'risk':frontier_std_with_RF, 'w': weights_frintier} )
w = pd.DataFrame(results['w'].values.tolist(), columns=stocks)
results = results.join(w).drop('w' , axis = 1).round(3)

#results.to_csv('results.csv')
=======

results = pd.DataFrame({'Return' :frontier_returns, 'Minimum_Risk':frontier_std_with_RF, 'w': weights_frintier_with_RF} )
w = pd.DataFrame(results['w'].values.tolist(), columns=stocks)
results = results.join(w).drop('w' , axis = 1).round(3)

results.to_csv('out/results.csv')
>>>>>>> Stashed changes
