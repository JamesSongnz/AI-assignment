# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:42:20 2019

@author: lenie
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
matplotlib.style.use('ggplot')

np.random.seed(1)
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 10000),
    'x2': np.random.normal(5, 3, 10000),
    'x3': np.random.normal(-5, 5, 10000)
})


# Use StandardScaler
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])


# Plot and visualize
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)

ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)

plt.show()



##================ with currency data
dataset = pd.read_csv("data_1999-2018_csv.CSV")

feature = dataset.iloc[:,[1,5,8]].values

scaler2 = StandardScaler()

#scaler2.fit(feature)
#InputsTrainStd = scaler2.transform(feature)

InputsTrainStd = scaler.fit_transform(feature)


feature = pd.DataFrame(feature, columns=['NZD/USD', 'NZD/EUR', 'NZD/CNY'])
InputsTrainStd = pd.DataFrame(InputsTrainStd, columns=['NZD/USD', 'NZD/EUR', 'NZD/CNY'])

# Plot and visualize
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 10))

ax1.set_title('Before Scaling')
sns.kdeplot(feature['NZD/USD'], ax=ax1)
sns.kdeplot(feature['NZD/EUR'], ax=ax1)
sns.kdeplot(feature['NZD/CNY'], ax=ax1)
#sns.kdeplot(feature, ax=ax1)


ax2.set_title('After Standard Scaler')
sns.kdeplot(InputsTrainStd['NZD/USD'], ax=ax2)
sns.kdeplot(InputsTrainStd['NZD/EUR'], ax=ax2)
sns.kdeplot(InputsTrainStd['NZD/CNY'], ax=ax2)


plt.show()