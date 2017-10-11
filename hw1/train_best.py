# #!/bin/bash
# python train.py ./data/train103-78.csv ./result/res103-78.csv
# python test.py ./data/test.csv ./result/res103-78.csv

import sys
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import regularizers

# define base model
def baseline_model():
	# Create model.
	model = Sequential()
	model.add(Dense(18, input_dim = 18, \
		kernel_initializer = 'normal', activation = 'relu'))
	model.add(Dense(1, kernel_initializer = 'normal'))
	# Compile model
	model.compile(loss = 'mean_squared_error', optimizer = 'adam')
	return model

if __name__ == "__main__":
	# Load input file path from argument.
	INPUT_FILE_PATH = sys.argv[1]

	# Load output file path from argument.
	OUTPUT_FILE_PATH = sys.argv[2]

	# Define constants.
	# Order of features in csv.
	ORDER_AMP_TEMP = 0          # 大氣溫度
	ORDER_CH4 = 1               # 甲烷
	ORDER_CO = 2                # 一氧化碳
	ORDER_NMHC = 3              # 非甲烷碳氫化合物
	ORDER_NO = 4                # 一氧化氮
	ORDER_NO2 = 5               # 二氧化氮
	ORDER_NOx = 6               # 氮氧化物
	ORDER_O3 = 7                # 臭氧
	ORDER_PM10 = 8              # 懸浮微粒
	ORDER_PM25 = 9              # 細懸浮微粒
	ORDER_RAINFALL = 10         # 雨量
	ORDER_RH = 11               # 相對溼度
	ORDER_SO2 = 12              # 二氧化硫
	ORDER_THC = 13              # 總碳氫合物
	ORDER_WD_HR = 14            # 風向小時值(以整個小時向量平均)
	ORDER_WIND_DIREC = 15       # 風向(以每小時最後10分鐘向量平均)
	ORDER_WIND_SPEED = 16       # 風速(以每小時最後10分鐘算術平均)
	ORDER_WS_HR = 17            # 風速小時值(以整個小時算術平均)

	# Total features in a day.
	DAY_TOTAL_FEATURE = 18
	
	# Order of time in csv.
	ORDER_HOUR = 3

	# Features to train.
	FEATURE_TRAIN_LIST = np.array(
		# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
		# [5, 7, 8, 9, 10, 12, 15, 16, 17])
		# [8, 9])
		[9])

	# How many features to train.
	FEATURE_NUMBER = len(FEATURE_TRAIN_LIST)
	print ("How many features to train: %d" %(FEATURE_NUMBER))
	print ("FEATURE_TRAIN_LIST: ", FEATURE_TRAIN_LIST)
	
	# How many hours to train.
	WINDOW_SIZE = 9

	# Total shift amount of window.
	WINDOW_SHIFT_AMOUNT = 15
	
	# Total Days to train.
	TRAIN_DAYS = 303
	# Day offset to train.
	TRIAN_DAY_OFFSET = 0
	print ("Training: Total Days: %d, Day offset: %d" %(TRAIN_DAYS, TRIAN_DAY_OFFSET))

	# Total Days to vaild.
	VALID_DAYS = 80
	# Day offset to vaild.
	VALID_DAY_OFFSET = 80
	print ("Testing: Total Days: %d, Day offset: %d" %(VALID_DAYS, VALID_DAY_OFFSET))

	# Read csv using pandas.
	trainCSV = pandas.read_csv(INPUT_FILE_PATH, encoding = 'Big5')

	# y_data = b + w * x_data
	# Data set for training.
	x_data = np.zeros((WINDOW_SIZE * FEATURE_NUMBER, WINDOW_SHIFT_AMOUNT * TRAIN_DAYS))
	y_data = np.zeros(WINDOW_SHIFT_AMOUNT * TRAIN_DAYS)
	
	# Data set for validation.
	x_valid_data = np.zeros((WINDOW_SIZE * FEATURE_NUMBER, WINDOW_SHIFT_AMOUNT * VALID_DAYS))
	y_valid_data = np.zeros(WINDOW_SHIFT_AMOUNT * VALID_DAYS)

	# Load training data set from csv.
	# Look at each day.
	for day in range(TRAIN_DAYS):
		# Look at each window.
		for n in range(WINDOW_SHIFT_AMOUNT):
			# Look through a range of hours.
			for hour in range(WINDOW_SIZE):
				# Look through each feature.
				for feature in range(FEATURE_NUMBER):
					# RAINFALL is string type.
					if (FEATURE_TRAIN_LIST[feature] == ORDER_RAINFALL):
						value = trainCSV.iloc[(day + TRIAN_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n]
						if (value == 'NR'):
							x_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = 0
						else:
							x_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = \
							float(trainCSV.iloc[(day + TRIAN_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n])  
					# Else data is in float type.
					else:
						x_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = \
						float(trainCSV.iloc[(day + TRIAN_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n])
			# Load training data set from csv.
			y_data[day * WINDOW_SHIFT_AMOUNT + n] = \
			float(trainCSV.iloc[(day + TRIAN_DAY_OFFSET) * DAY_TOTAL_FEATURE + ORDER_PM25, ORDER_HOUR + WINDOW_SIZE + n])

	# Load validation data set from csv.
	# Look at each day.
	for day in range(VALID_DAYS):
		# Look at each window.
		for n in range(WINDOW_SHIFT_AMOUNT):
			# Look through a range of hours.
			for hour in range(WINDOW_SIZE):
				# Look through each feature.
				for feature in range(FEATURE_NUMBER):
					# RAINFALL is string type.
					if (FEATURE_TRAIN_LIST[feature] == ORDER_RAINFALL):
						value = trainCSV.iloc[(day + VALID_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n]
						if (value == 'NR'):
							x_valid_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = 0
						else:
							x_valid_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = \
							float(trainCSV.iloc[(day + VALID_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n])  
					# Else data is in float type.
					else:
						x_valid_data[feature * WINDOW_SIZE + hour][day * WINDOW_SHIFT_AMOUNT + n] = \
						float(trainCSV.iloc[(day + VALID_DAY_OFFSET) * DAY_TOTAL_FEATURE + FEATURE_TRAIN_LIST[feature], ORDER_HOUR + hour + n])
			# Load validation data set from csv.
			y_valid_data[day * WINDOW_SHIFT_AMOUNT + n] = \
			float(trainCSV.iloc[(day + VALID_DAY_OFFSET) * DAY_TOTAL_FEATURE + ORDER_PM25, ORDER_HOUR + WINDOW_SIZE + n])

	# Order of fitting term.
	FITTING_TERM_ORDER = 2
	print ("Fitting term order: %d" %(FITTING_TERM_ORDER))
	# Add square term.
	if (FITTING_TERM_ORDER == 2):
		x_data = np.concatenate((x_data, x_data ** 2), axis = 0)
		x_valid_data = np.concatenate((x_valid_data, x_valid_data ** 2), axis = 0)
	
	# Data format to keras.
	x_data = np.transpose(x_data)
	x_valid_data = np.transpose(x_valid_data)

	# Define Epochs and batch size.
	epochs = 100
	batchSize = 16
	print ("Epochs: %d, batch_size = %d" %(epochs, batchSize))

	# # evaluate model with standardized dataset
	# estimator = KerasRegressor(build_fn = baseline_model, \
	# 	epochs = epochs, batch_size = batchSize, verbose = True)
	# # fix random seed for reproducibility
	# kfold = KFold(n_splits = 10, shuffle = True, random_state = np.random.seed(7))
	# results = cross_val_score(estimator, x_data, y_data, cv = kfold)
	# print("\nResults: %.2f (%.2f) MSE" % (results.mean(), results.std()))

	# # Fit the model
	# estimator.fit(x_data, y_data)
	# # Save model to HDF5 file.
	# estimator.model.save('model_f1o2l2k10.h5')

	# Create Model.
	model = baseline_model()
	# Fit the model
	model.fit(x_data, y_data, \
		epochs = epochs, batch_size = batchSize, \
		shuffle=True, verbose = True)
	# Save model to HDF5 file.
	model.save('model_f1o2l2.h5')