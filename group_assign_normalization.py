
"""
Spyder Editor

This is a temporary script file.
"""


# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt




from pandas import Series
from sklearn.preprocessing import MinMaxScaler

# set seed for reproducibility
#np.random.seed(0)

##  INFO 813
nzd_usd = np.zeros((1,5023))
# read in all our data
dataset = pd.read_csv("data_1999-2018_csv.CSV")

feature = dataset.iloc[:,[1]].values



## === use std scaler

from sklearn.preprocessing import StandardScaler
Scaler = MinMaxScaler()
Scaler.fit(feature)
InputsTrainStd = Scaler.transform(feature)

# plot both together to compare
fig, ax=plt.subplots(1,2, figsize=(15, 10))
sns.distplot(feature, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(InputsTrainStd, ax=ax[1])
ax[1].set_title("MinMax Scaled data")

'''
## ============================================
##  use sklewarn

original_data = nzd_usd.tolist()

np_currencies  = np.array(currencies)

nzd_usd = np_currencies[:, 1:2]
nzd_usd = np.reshape(nzd_usd, 5023)


series = Series(original_data)
print(series)
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print
normalized = scaler.transform(values)
print(normalized)
# inverse transform and print
inversed = scaler.inverse_transform(normalized)
print(inversed)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized, ax=ax[1])
ax[1].set_title(" SKlearn :  Normalized data")


## ============================================ 
# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(nzd_usd, columns = [0])
##X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
##X_scaled = X_std * (max - min) + min   MinMaxScaler(feature_range=(0, 1), copy=True)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")

print("hello")

# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data) 

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")
'''