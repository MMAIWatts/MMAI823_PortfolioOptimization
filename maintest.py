import pandas as pd
import datetime
import os
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
    print(d.head())

# data_master = data_clean[0].join(data_clean[1], on='Date', rsuffix='_1')
data_master = data_clean[0].join(data_clean[1], lsuffix='_0', rsuffix='_1')
print(data_master.info())
print(data_master.head())

data_returns = pd.DataFrame()
# calculate daily returns
for i in range(len(data_clean)):
    data_returns['daily_return_' + str(i)] = d['Adj Close_' + str(i)].pct_change()

print(data_returns.head())
