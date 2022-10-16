import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.losses import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
file_dic = r'C:\Users\Desktop\2022年E题\数据集\基本数据/'
file_name = r'附件3、土壤湿度2022—2012年.xlsx'
# dataset_train = pd.read_excel(file_dic + file_name, usecols=['10cm湿度(kg/m2)'], sheet_name='sheet1')
dataset_train = pd.read_excel(file_dic + file_name, usecols=['10cm湿度(kg/m2)'], sheet_name='sheet1')
# dataset_train = dataset_train.sort_values(by='Date').reset_index(drop=True)
training_set = dataset_train.values
print(dataset_train.shape)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# 每条样本含60个时间步，对应下一时间步的标签值
X_train = []
y_train = []
for i in range(6, 93):
    X_train.append(training_set_scaled[i - 6:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(y_train.shape)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
from keras.layers import Dropout
# print(X_train.shape[1])
# 初始化顺序模型
regressor = Sequential()
# 定义输入层及带5个神经元的隐藏层
regressor.add(SimpleRNN(units=15, input_shape=(X_train.shape[1], 1)))
# 定义线性的输出层
regressor.add(Dense(units=1, activation='relu'))
# 模型编译：定义优化算法adam， 目标函数均方根MSE
regressor.compile(optimizer='sgd', loss='mean_squared_error')
# 模型训练
history = regressor.fit(X_train, y_train, epochs=40, batch_size=20, validation_split=0.1)
regressor.summary()
ax1 = plt.subplot(121)
ax1.plot(history.history['loss'], c='blue', label='训练集损失')  # 蓝色线训练集损失
ax1.plot(history.history['val_loss'], c='red', label='验证集损失')  # 红色线验证集损失
plt.ylabel('值')
plt.xlabel('迭代次数')
plt.legend()
# 测试数据
# dataset_test = pd.read_csv('./data/tatatest.csv')
# dataset_test = pd.read_excel(file_dic + r'附件3、土壤湿度2022—2012年test.xlsx', usecols=['10cm湿度(kg/m2)'], sheet_name='sheet1')
dataset_test = pd.read_excel(file_dic + r'附件3、土壤湿度2022—2012年test.xlsx', usecols=['10cm湿度(kg/m2)'], sheet_name='sheet1')
real_value = dataset_test['10cm湿度(kg/m2)'].values

dataset_total = pd.concat((dataset_train['10cm湿度(kg/m2)'], dataset_test['10cm湿度(kg/m2)']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# 提取测试集
X_test = []
for i in range(6, 30):
    X_test.append(inputs[i - 6:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 模型预测
predicted_value = regressor.predict(X_test)
# 逆归一化
predicted_value = sc.inverse_transform(predicted_value)
# 模型评估
# trainScore = math.sqrt(mean_squared_error(predicted_value[0], trainPredict[:, 0]))
print('预测与实际差异MSE', sum(pow((predicted_value - real_value), 2)) / predicted_value.shape[0])
print('预测与实际差异MAE', sum(abs(predicted_value - real_value)) / predicted_value.shape[0])

val = regressor.predict([[[X_test[-5:]]]])
val = sc.inverse_transform(val)
blo = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09']
resl = []
valu = []
for i in range(len(blo)):
    resl.append(blo[i] + str(val))
    val = sc.fit_transform(val)
    # print(val.shape)
    val = regressor.predict([val[-5:]])
    val = sc.inverse_transform(val)
    valu.append(val[-1][0])
# 预测与实际差异的可视化
ax2 = plt.subplot(122)
ax2.plot(real_value, color='red', label='真实值')
valu = np.array([valu]).reshape(-1, 1)
predicted_value = np.concatenate((predicted_value, valu), axis=0)
ax2.plot(predicted_value, color='blue', label='预测值')
# plt.title('TATA Stock Price Prediction')
plt.xlabel('迭代次数')

plt.legend()
plt.savefig('Q3/' + 'q63.jpg', dpi=300)
plt.show()
