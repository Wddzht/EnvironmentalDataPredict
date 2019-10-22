# coding=utf-8
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# Decision Tree Regression Portland and Fuzhou2017010618
# 读取数据，划分训练集和测试集
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 6  # 污染物的个数

file_micro = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt")
file_nation = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_prediction = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\ANN_prediction.pare.txt")
data_micro = []
data_nation = []
data_prediction = []
# 读取数据
for line in file_micro:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    data_micro.append(_list)
for line in file_nation:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    data_nation.append(_list)
for line in file_prediction:
    line=line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    data_prediction.append(_list)

fit_output=np.zeros((len(data_prediction), N), dtype=np.double)
# 多元线性回归
linear_model = DecisionTreeRegressor(max_depth=8)

for i in range(0, N):
    X_train = np.array(data_micro)[:,0:N] # 只用前6个数据 反而能增加准确度
    X_test = np.array(data_prediction)[:,0:N]
    y_train = []
    for item in data_nation:
        y_train.append(item[i])

    # 模型训练
    linear_model.fit(X_train, y_train)
    # 测试
    y_pre = linear_model.predict(X_test)
    for r in range(len(y_pre)):
        fit_output[r][i]=y_pre[r]

    # Plot the results
    plt.figure()
    x = range(0, 1000)
    plt.plot(x, y_train[:1000], color="b", label="实际值", linewidth=1.5)
    plt.plot(x, y_pre[:1000], color="r", label="线性回归预测值", linewidth=1.5) # (np.array(X_train)[:,i])[-1000:]

    plt.xlabel("time/h")
    plt.ylabel("value")
    plt.title("数据预测-" + item_name[i])
    plt.legend()
    plt.show()

output_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit\ANN_prediction.pare.fit.txt", 'w')
for item in fit_output:
    for val in item:
        output_file.write('{:.4}\t'.format(val))
    output_file.write('\n')
output_file.close()
