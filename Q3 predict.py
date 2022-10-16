import numpy
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


# https://blog.csdn.net/zyxhangiian123456789/article/details/87458140
# https://blog.csdn.net/LaoChengZier/article/details/90511968
# https://deephub.blog.csdn.net/article/details/122425490
# https://blog.csdn.net/weixin_52855810/article/details/112982229
# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]  # 用look_back个样本来预测一个数据
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


iq = 0


def getValue(data, name):
    dataset = data.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    dataset = dataset.reshape(-1, 1)
    # 数据处理，归一化至0~1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 划分训练集和测试集
    train_size = 27
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # train, test = dataset[:30, :], dataset[:30, :]
    # 创建测试集和训练集
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)  # 单步预测
    testX, testY = create_dataset(test, look_back)

    # 调整输入数据的格式
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))  # （样本个数，1，输入的维度）
    testX = numpy.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))

    # 创建LSTM神经网络模型
    model = Sequential()
    model.add(LSTM(120, unit_forget_bias=True, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(100))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    # model.add(LSTM(120, input_shape=(trainX.shape[1], trainX.shape[2])))  # 输入维度为1，时间窗的长度为1，隐含层神经元节点个数为120
    # model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.summary()
    # 绘制网络结构
    # plot_model(model, to_file='model.png', show_shapes=True)

    history = model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0, validation_data=(testX, testY))

    # 预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # print(trainPredict.shape, testPredict.shape)

    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # 计算得分
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % testScore)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    val = model.predict([[[testX[-1]]]])
    val = scaler.inverse_transform(val)
    blo = ['A', 'B', 'C']
    resl = []
    for i in range(len(blo)):
        resl.append(blo[i] + str(val))
        val = scaler.fit_transform(val)
        val = model.predict([[[val]]])
        val = scaler.inverse_transform(val)
    # 绘
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    plt.figure()
    plt.plot(scaler.inverse_transform(dataset), label='真实值', color='b')
    plt.plot(trainPredictPlot, label='训练值', color='black')
    plt.plot(testPredictPlot, label='预测值', color='r')
    # plt.title(name + '拟合曲线')
    plt.ylabel('值')
    plt.legend()
    plt.savefig('picture/lstm/' + name + str(iq), bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()

    print(resl)
    return resl


def load_data(data):
    cho = ['NG', 'LGI', 'MGI', 'HGI']
    col = ['SOC土壤有机碳', 'SIC土壤无机碳', '全氮N']
    results = []
    for i in cho:
        result = []
        for j in range(data.shape[0]):
            if data['放牧强度（intensity）'][j] == i:
                result.append(data[j:j + 1][col])
        results.append(result)
    results = np.array(results).reshape((4, 33, -1))
    return results


if __name__ == '__main__':
    # 加载数据
    file_dic = r'C:\Users\Desktop\2022年E题\数据集\监测点数据\附件14：不同放牧强度土壤碳氮监测数据集/'
    file_name = r'不同放牧强度土壤碳氮监测数据集.xlsx'
    cols = ['放牧强度（intensity）', 'SOC土壤有机碳', 'SIC土壤无机碳', '全氮N']
    dataframe = read_excel(file_dic + file_name, usecols=cols)
    dataframe = load_data(dataframe)
    values = []
    for i in range(len(dataframe)):
        res = pd.DataFrame(dataframe[i], columns=cols[1:])
        for j in range(len(cols[1:])):
            iq += 1
            temp = res[cols[1 + j]]
            name = cols[1 + j] + str(i) + str(j)
            val = getValue(temp, name)
            for k in val:
                values.append(name + k)
    print(values)
