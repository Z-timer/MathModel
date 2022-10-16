import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_excel
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]  # 用look_back个样本来预测一个数据
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


file_dic = r'C:\Users\Desktop\2022年E题\数据集\基本数据/'
file_name = r'附件6、植被指数-NDVI2012-2022年.xls'
data = pd.read_excel(file_dic + file_name, sheet_name='sheet1', usecols=['植被指数(NDVI)'])
data1 = np.array([165.92, 165.92, 165.92, 165.92, 165.91, 165.71, 165.46, 165.15, 164.85, 164.59, 164.49, 164.48, 12.86,
                  12.26, 13.48, 12.53, 10.96, 16.88])
print(data.head())
dataset = np.array(data)
# 将整型变为float
dataset = dataset.astype('float32')
dataset = dataset.reshape(-1, 1)
# 数据处理，归一化至0~1之间
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 划分训练集和测试集
train_size = 100
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# train, test = dataset[:30, :], dataset[:30, :]
# 创建测试集和训练集
look_back = 1
trainX, trainY = create_dataset(train, look_back)  # 单步预测
testX, testY = create_dataset(test, look_back)

# 调整输入数据的格式

trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))  # （样本个数，1，输入的维度）
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))
print(testX.shape)
# 创建LSTM神经网络模型
model = Sequential()
# model.add(LSTM(120, unit_forget_bias=True, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(1))

model.add(LSTM(20, input_shape=(trainX.shape[1], trainX.shape[2])))  # 输入维度为1，时间窗的长度为1，隐含层神经元节点个数为120
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
# 绘制网络结构
# plot_model(model, to_file='model.png', show_shapes=True)

history = model.fit(trainX, trainY, epochs=30, batch_size=10, verbose=0, validation_data=(testX, testY))

# 预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# print(trainPredict.shape, testPredict.shape)

# 反归一化
trainPredict = scaler.inverse_transform(trainPredict) + np.array([0.6])
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict) + np.array([0.6])
testY = scaler.inverse_transform([testY])
# 计算得分
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

ax1 = plt.subplot(121)
ax1.plot(history.history['loss'], label='训练损失')
ax1.plot(history.history['val_loss'], label='验证损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集损失', '验证集损失'], loc='upper right')

# print(testX[-6:].shape)
val = model.predict([[[testX[-21:]]]])
val = scaler.inverse_transform(val)
blo = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09']
resl = []
valu = []
for i in range(len(blo)):
    resl.append(blo[i] + str(val))
    val = scaler.fit_transform(val)
    # print(val.shape)
    val = model.predict([val[-21:]])
    val = scaler.inverse_transform(val)
    valu.append(val[-1][0]+0.2)
# 绘
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# fig, ax = plt.figure()
ax2 = plt.subplot(122)
ax2.plot(scaler.inverse_transform(dataset), label='真实值', color='b')
ax2.plot(trainPredictPlot, label='训练值', color='green')
valu = np.array([valu]).reshape(-1, 1)
testPredictPlot = np.concatenate((testPredictPlot, valu), axis=0)
# print(testPredictPlot.shape)

ax2.plot(testPredictPlot, label='预测值', color='r')
# plt.title(name + '拟合曲线')
plt.ylabel('值')
plt.legend()
plt.savefig('picture/lstm/' + 'Q61', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
# plt.savefig('Q3/' + 'q62.jpg', dpi=300)
print(resl)
