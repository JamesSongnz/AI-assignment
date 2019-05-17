
from keras.models import load_model
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load model from single file
model = load_model('lstm_model.h5')


dataframe = pd.read_csv("1999-2019.CSV") 
feature = dataframe.iloc[:,[1]].values 
scaler = MinMaxScaler(feature_range=(0, 1)) 
dataset = scaler.fit_transform(feature) 

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
	
# split into train and test sets
trainSize = int(len(dataset) * 0.60)
validataionSize = int(len(dataset) * 0.20)
testSize = len(dataset) - trainSize - validataionSize

train, validation, test = dataset[0:trainSize,:], dataset[trainSize:trainSize+validataionSize,:], dataset[trainSize+validataionSize:len(dataset),:]
print(len(train), len(validation), len(test))



look_back = 8


trainX, trainY = create_dataset(train, look_back)
validationX, validationY = create_dataset(validation, look_back)
testX, testY = create_dataset(test, look_back)

print('trainX = ' , trainX)
print('trainY = ' ,trainY)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
validationX = numpy.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# make predictions
trainPredict = model.predict(trainX)
validationPredict = model.predict(validationX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
validationPredict = scaler.inverse_transform(validationPredict)
validationY = scaler.inverse_transform([validationY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.10f RMSE' % (trainScore))
validationScore = math.sqrt(mean_squared_error(validationY[0], validationPredict[:,0]))
print('VAlidation Score: %.10f RMSE' % (validationScore))



testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.10f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#validation
validationPredictPlot = numpy.empty_like(dataset)
validationPredictPlot[:, :] = numpy.nan
validationPredictPlot[len(trainPredict)+(look_back)+1:len(validationPredict)+len(trainPredict)+look_back+1, :] = validationPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(validationPredict)+len(trainPredict)+(look_back*3)+2:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(validationPredictPlot)

plt.plot(testPredictPlot)
plt.legend()
plt.show()


