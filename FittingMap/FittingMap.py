# coding=utf-8
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# Decision Tree Regression Portland and Fuzhou2017010618
# 读取数据，划分训练集和测试集
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 6  # 污染物的个数

file_micro = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt")
file_nation = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")

data_micro = []
data_nation = []
for line in file_micro:
    line = line.strip()
    if not line:
        break
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    data_micro.append(_list)
for line in file_nation:
    line = line.strip()
    if not line:
        break
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    data_nation.append(_list)

for model in ['DecisionTree','GBDT','LinearRegression','RFR']:
    file_prediction = open(r".\data\fz_prediction\{:}.pare.txt".format(model))
    output_path = r".\data\fz_prediction_fit\小输入预测\{:}.pare.fit.txt".format(model)

    data_prediction = []
    for line in file_prediction:
        line = line.strip()
        if not line:
            break
        _list = []
        for i in line.split('\t'):
            _list.append(float(i))
        data_prediction.append(_list)

    fit_output = np.zeros((len(data_prediction), N), dtype=np.double)

    for i in range(0, N):
        linear_model = LinearRegression()
        X_train = []
        X_test = []
        # 控制拟合参数 反而能增加准确度
        if item_name[i] == 'O3':  # T NO2 H
            for item, pre in zip(data_micro, data_prediction):
                X_train.append([item[1], item[3]])
                X_test.append([pre[1], pre[3]])
        elif item_name[i] == 'NO2':  # T NO2 H
            for item, pre in zip(data_micro, data_prediction):
                X_train.append([item[1], item[3]])
                X_test.append([pre[1], pre[3]])
        else:
            for item, pre in zip(data_micro, data_prediction):
                X_train.append([item[i],item[0],item[1],item[2],item[3],item[4],item[5]])
                X_test.append([pre[i],pre[0],pre[1],pre[2],pre[3],pre[4],pre[5]])

        y_train = []
        for item in data_nation:
            y_train.append(item[i])

        # max_data = (X_train[0])[:]
        # min_data = (X_train[0])[:]
        # for k in range(len(X_train)):
        #     for j in range(len(X_train[0])):
        #         if X_train[k][j] > max_data[j]:
        #             max_data[j] = X_train[k][j]
        #         elif X_train[k][j] < min_data[j]:
        #             min_data[j] = X_train[k][j]
        #         if X_test[k][j] > max_data[j]:
        #             max_data[j] = X_test[k][j]
        #         elif X_test[k][j] < min_data[j]:
        #             min_data[j] = X_test[k][j]
        # for k in range(len(X_train)):
        #     for j in range(len(X_train[0])):
        #         if max_data[j] - min_data[j] == 0:
        #             X_train[k][j] = X_train[k][j] - min_data[j]
        #             X_test[k][j] = X_test[k][j] - min_data[j]
        #         else:
        #             X_train[k][j] = (X_train[k][j] - min_data[j]) / (max_data[j] - min_data[j])
        #             X_test[k][j] = (X_test[k][j] - min_data[j]) / (max_data[j] - min_data[j])

        # 模型训练
        linear_model.fit(X_train, y_train)
        # 测试
        y_pre = linear_model.predict(X_test)
        y_pre_train = linear_model.predict(X_train)
        for r in range(len(y_pre)):
            fit_output[r][i] = y_pre[r]

        # Plot the results
        plt.figure()
        x = range(0, 1500)
        plt.plot(x, y_pre[:1500], label="预测结果线性回归拟合", linewidth=1.5)  # (np.array(X_train)[:,i])[-1000:]

        plt.plot(x, y_pre_train[:1500], label="实际值线性回归拟合", linewidth=1.5)
        plt.plot(x, y_train[:1500], 'r--', label="实际值", linewidth=1.5)
        plt.xlabel("time/h")
        plt.ylabel("value")
        plt.title("微测站数据拟合-" + item_name[i])
        plt.legend()
        print(
            '{:} 拟合损失R2:{:.2f} 预测R2:{:.2f}'.format(item_name[i], r2_score(y_train, y_pre_train), r2_score(y_train, y_pre)))
        #plt.show()

    output_file = open(output_path, 'w')
    for item in fit_output:
        for val in item:
            output_file.write('{:}\t'.format(val))
        output_file.write('\n')
    output_file.close()
