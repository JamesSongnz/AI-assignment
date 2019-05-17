# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:42:20 2019

@author: James Song PGDIT at AIS
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
matplotlib.style.use('ggplot')


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