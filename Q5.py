import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_data():
    file14_dic = r'C:\Users\Desktop\2022年E题\数据集\监测点数据\附件14：不同放牧强度土壤碳氮监测数据集/'
    file14_name = r'不同放牧强度土壤碳氮监测数据集.csv'
    file15_dic = r'C:\Users\Desktop\2022年E题\数据集\监测点数据\附件15：草原轮牧放牧样地群落结构监测数据集（2016年6月-2020年9月）。/'
    file15_name = r'内蒙古自治区锡林郭勒盟典型草原轮牧放牧样地群落结构监测数据集（201.xlsx'
    data15 = pd.read_excel(file15_dic + file15_name, sheet_name='Sheet1')

    data14 = pd.read_csv(file14_dic + file14_name, encoding='unicode_escape')
    print(data15.columns)
    print(data14.columns)
    X = data14[['SOC', 'SIC', 'N', 'jyl']]
    X = np.array(X)
    Y = data14['intensity']
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = RandomForestRegressor(max_depth=10, n_estimators=100)
    print(X_train.shape, Y_train.shape)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, Y_test)
    print(' 得分:' + str(score))
    plt.figure()
    x = np.arange(0, 17)
    plt.plot(x, Y_test, color='#E0A97C', label="TRUE")
    plt.plot(x, y_pred, color='#889BB7', label="PREDICT")
    plt.legend()
    plt.show()
    mse = mean_squared_error(Y_test, y_pred)
    mae = mean_absolute_error(Y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))  # RMSE就是对MSE开方即可
    r2 = r2_score(Y_test, y_pred)
    print('mse: ', mse, 'mae: ', mae, 'rmse: ', rmse, 'r2: ', r2)
    model.predict([])


if __name__ == '__main__':
    load_data()
