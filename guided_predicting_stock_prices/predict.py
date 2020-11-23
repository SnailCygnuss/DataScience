import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

sp500 = pd.read_csv('sphist.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500 = sp500.sort_values(by='Date', ascending=True)
print(sp500.info())
print(sp500.head())

close_prices = {}
close_prices['day_5'] = []
close_prices['day_30'] = []
close_prices['day_365'] = []
close_prices['std_5'] = []
close_prices['std_365'] = []
close_prices['vol_5'] = []
close_prices['vol_365'] = []

first_index = sp500.index[0]

for d, _ in sp500.iterrows():
# for d in range(16584, 16590):
    # Index for 5 day interval
    d5_1   = d + 5
    d5_2   = d + 1
    # Index for 30 day interval
    d30_1  = d + 30
    d30_2  = d + 1
    # Index for 365 day interval
    d365_1 = d + 365
    d365_2 = d + 1
    
    day_5_data = sp500.loc[d5_1:d5_2, 'Close']
    day_30_data = sp500.loc[d30_1:d30_2, 'Close']
    day_365_data = sp500.loc[d365_1:d365_2, 'Close']
    
    vol_5_data = sp500.loc[d5_1:d5_2, 'Volume']
    vol_365_data = sp500.loc[d365_1:d365_2, 'Volume']
    
    if d365_1 > first_index:
        close_prices['day_5'].append(0)
        close_prices['day_30'].append(0)
        close_prices['day_365'].append(0)
        close_prices['std_5'].append(0)
        close_prices['std_365'].append(0)
        close_prices['vol_5'].append(0)
        close_prices['vol_365'].append(0)
    # elif d30_1 > first_index:
    #     close_prices['day_5'].append(day_5_data.mean())
    #     close_prices['day_30'].append(0)
    #     close_prices['day_365'].append(0)
    #     close_prices['std_5'].append(day_5_data.std())
    #     close_prices['std_365'].append(0)
    # elif d365_1 > first_index:
    #     close_prices['day_5'].append(day_5_data.mean())
    #     close_prices['day_30'].append(day_30_data.mean())           
    #     close_prices['day_365'].append(0)
    #     close_prices['std_5'].append(day_5_data.std())
    #     close_prices['std_365'].append(0)
    else:
        close_prices['day_5'].append(day_5_data.mean())
        close_prices['day_30'].append(day_30_data.mean())
        close_prices['day_365'].append(day_365_data.mean())
        close_prices['std_5'].append(day_5_data.std())
        close_prices['std_365'].append(day_365_data.std())
        close_prices['vol_5'].append(vol_5_data.mean())
        close_prices['vol_365'].append(vol_365_data.mean())

        
close_prices = pd.DataFrame(close_prices, index=range(first_index, -1, -1))

sp500 = pd.concat([sp500, close_prices], axis=1)
sp500['avg_ratio_5_365'] = sp500['day_5'] / sp500['day_365']
sp500['std_ratio_5_365'] = sp500['std_5'] / sp500['std_365']
sp500['avg_vol_ratio_5_365'] = sp500['vol_5'] / sp500['vol_365']
sp500['month'] = pd.DatetimeIndex(sp500['Date']).year
# Replace 0 and inf with NaN
sp500 = sp500.replace([0, np.inf], np.nan)

# Remove rows before 1951-01-03 will be removed as columns day_365
# and std_365 have empty values.
sp500 = sp500[sp500['Date'] > datetime(year=1951, month=1, day=2)]
# There are few more rows with 0 values for these columns
sp500 = sp500.dropna(axis=0)
print(sp500.head())

# Train and test DataFrame
train = sp500[sp500['Date'] < datetime(year=2013, month=1, day=1)]
test = sp500[sp500['Date'] >= datetime(year=2013, month=1, day=1)]

linear_model = LinearRegression()
cols = ['day_5', 'day_30', 'day_365', 'std_5', 'std_365',
       'avg_ratio_5_365', 'std_ratio_5_365', 'avg_vol_ratio_5_365',
       'month']
X = train[cols]
y = train['Close']
linear_model.fit(X, y)
predictions = linear_model.predict(test[cols])

mae = np.mean(np.abs(predictions - test['Close']))
print('MAE = {}'.format(mae))