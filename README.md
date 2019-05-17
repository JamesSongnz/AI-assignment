# AI-assignment
AI assignment 2019 for Auckland Inst. 

This builds LSTM model to predict exchange rate of NZD/USD during 1999-2019. 

* use Keras library
* 3 layer LSTM model 
* time series dataset
* preprocessing : MinMaxScaling


# introduction
We chose time series data for exchange rate between NZD and USD. Our dataset includes daily exchange rate from 1999 to current date, 5098 observations in total. 
The Topic is prediction of the next day of NZD/USD currency price based on past data set with range of 1999~ 2109 provided by the bank. We used the graph visualisiation technique which is suitable way for time series data to show its trend corresponding to date. MinMax Scaling transform is adopted to change input value into range of 0 to 1. It helps that machine learning algorithm yields better performance.  

# PreProcessing
-Slide Time Window 
-Min-Max Scaler 

# Model 
LSTM : LSTM is a special model of recurrent neural networks (RNN). It aims to solve gradient vanishing problem and gradient exploding problem in the process of long sequence training. 

# dataset

Split data set into 3 parts (train, validation, and test) 

Train(60%) 
Validation(20%) 
Test(20%) 

Walk-forward : 4fold 
 
# Loss & Optimzer
Loss function: RMSE (root mean squared error)  
Optimizer: Adam  

# Initial parameters for our model  

. The number of stacks of LSTM: 3 

. The number of LSTM dimension:  32  

. Loss function: RMSE 

. Optimizer: Adam 

. Batch size: 1 

. The number of epochs: 50 ~ 


# best result
* Walk-forward: Fold 4 with size of 50% of total dataset, trainset 90%, validation 10% 
* 3 layers with 32 dimension Epochs 55 Batch 16 

reference :  LSTM example blog site 
- Pant, N. (2017) Using recurrent neural networks (LSTMs). Retrieved from 
https://blog.statsbot.co/time-series-prediction-using-recurrent-neural-networks-lstms-807fa6ca7f 
