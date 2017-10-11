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
from keras.models import load_model

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
    ORDER_HOUR = 2

    # Features to test.
    FEATURE_TEST_LIST = np.array(
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        # [5, 7, 8, 9, 12, 16, 17])
        # [8, 9])
        [9])

    # How many features to test.
    FEATURE_NUMBER = len(FEATURE_TEST_LIST)
    print ("How many features to test: %d" %(FEATURE_NUMBER))
    print ("FEATURE_TEST_LIST: ", FEATURE_TEST_LIST)

    # How many hours to test.
    WINDOW_SIZE = 9

    # Total Days to test.
    TEST_DAYS = 240

    # Read test data from csv using pandas.
    testCSV = pandas.read_csv(INPUT_FILE_PATH, encoding = 'Big5', header=None)

    x_data = np.zeros((WINDOW_SIZE * FEATURE_NUMBER, TEST_DAYS))
    y_data = np.zeros(TEST_DAYS)
    # y_data = b + w * x_data
    # Look at each day.
    for day in range(TEST_DAYS):
            # Look through a range of hours.
            for hour in range(WINDOW_SIZE):
                # Look through each feature.
                for feature in range(FEATURE_NUMBER):
                    # RAINFALL is string type.
                    if (FEATURE_TEST_LIST[feature] == ORDER_RAINFALL):
                        value = testCSV.iloc[day * DAY_TOTAL_FEATURE + FEATURE_TEST_LIST[feature], ORDER_HOUR + hour]
                        if (value == 'NR'):
                            x_data[feature * WINDOW_SIZE + hour][day] = 0
                        else:
                            x_data[feature * WINDOW_SIZE + hour][day] = \
                            float(testCSV.iloc[day * DAY_TOTAL_FEATURE + FEATURE_TEST_LIST[feature], ORDER_HOUR + hour])  
                    # Else data is in float type.
                    else:
                        x_data[feature * WINDOW_SIZE + hour][day] = \
                        float(testCSV.iloc[day * DAY_TOTAL_FEATURE + FEATURE_TEST_LIST[feature], ORDER_HOUR + hour])
                    # x_data[feature * WINDOW_SIZE + hour][day] = \
                    # float(testCSV.iloc[day * DAY_TOTAL_FEATURE + ORDER_PM10 + feature, ORDER_HOUR + hour])

    # print (x_data)

    # Order of fitting term.
    FITTING_TERM_ORDER = 2
    print ("Fitting term order: %d" %(FITTING_TERM_ORDER))
    # Add square term.
    if (FITTING_TERM_ORDER == 2):
        x_data = np.concatenate((x_data, x_data ** 2), axis = 0)
    
    model = load_model('model_f1o2l2.h5')

    y_data = model.predict(np.transpose(x_data))
    y_data = y_data.reshape(-1)

    stringID = list()
    for n in range(TEST_DAYS):
        stringID.append('id_' + str(n))
    
    resultCSV_dict = {
        "id":       stringID,
        "value":    y_data
    }
    resultCSV = pandas.DataFrame(resultCSV_dict)
    resultCSV.to_csv(OUTPUT_FILE_PATH, index=False)