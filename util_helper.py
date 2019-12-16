import numpy as np
import pandas as pd
import pandas_datareader as pdr


# Takes in weights, returns array or return,standard deviation, sharpe ratio
def get_ret_std(weights, returns, covariance):
    """Calculate returns, standard deviation and sharpe ratio for a set of stocks.

    Arguments:
        returns -- a set of returns, as calculated from historic beta values
        weights -- a set of weights that define the portion allocated to each stock
        covariance -- a covariance matrix as calculated from the daily percent change in each stock

    Returns:
        array of:
        [0] -- Returns on each stock
        [1] -- Standard deviation of each stock
    """
    weights = np.array(weights)
    r = np.sum(returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

    return np.array([r, std])


# objective
def minimize_risk(weights, returns, covariance):
    """Help function for minimization function."""
    return get_ret_std(weights, returns, covariance)[1]


def get_stocks_yahoo(stocks, start, end, feature):
    """Retrieve a finance stock data from Yahoo finance.

    Arguments:
        stocks  -- list of ticker symbols
        start   -- start date of data to be retrieved
        end     -- end date of data to be retrieved
        feature -- feature column to be extracted (i.e. Adj_Close, Open, Close...)

    Returns:
        data    -- DataFrame of data retrieved
    """
    data = pd.DataFrame()
    for symbol in stocks:
        data[symbol] = pdr.get_data_yahoo(symbol, start=start, end=end)[feature]

    return data
