# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 01:16:23 2019

@author: James Song PGDIT at AIS
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

##================ with currency data
dataset = pd.read_csv("data_1999-2018_csv.CSV")

feature = dataset.iloc[:,[1,5,8]].values

# Use MinMaxScaler
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(feature)

feature = pd.DataFrame(feature, columns=['NZD/USD', 'NZD/EUR', 'NZD/CNY'])
scaled_df = pd.DataFrame(scaled_df, columns=['NZD/USD', 'NZD/EUR', 'NZD/CNY'])

# Plot and visualize
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 10))

ax1.set_title('Before Scaling')
sns.kdeplot(feature['NZD/USD'], ax=ax1)
sns.kdeplot(feature['NZD/EUR'], ax=ax1)
sns.kdeplot(feature['NZD/CNY'], ax=ax1)

ax2.set_title('After Min-Max Scaling')
sns.kdeplot(scaled_df['NZD/USD'], ax=ax2)
sns.kdeplot(scaled_df['NZD/EUR'], ax=ax2)
sns.kdeplot(scaled_df['NZD/CNY'], ax=ax2) 


plt.show()