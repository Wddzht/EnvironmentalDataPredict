# coding=utf-8
# 只针对PM25做拟合,拟合结果计算R2等误差,因为国测站没有温度,湿度指标.
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# Decision Tree Regression Portland and Fuzhou2017010618
# 读取数据，划分训练集和测试集
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 1  # 污染物的个数

file_micro = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_micro.pare.txt")
file_nation = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
file_prediction = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction\ANN_PM25_T_H_prediction.pare.txt")
data_micro = []
data_nation = []
data_prediction = []
# 读取数据
for line in file_micro:
    line = line.strip()
    if not line:
        continue
    data_micro.append(float(line.split('\t')[4]))
for line in file_nation:
    line = line.strip()
    if not line:
        continue
    data_nation.append(float(line.split('\t')[4]))
for line in file_prediction:
    line = line.strip()
    if not line:
        continue
    data_prediction.append(float(line.split('\t')[0]))  # 只取['PM25', 'TEMPERATURE', 'HUMIDITY'] 的 PM25

fit_output = np.zeros((len(data_prediction), N), dtype=np.double)
# 多元线性回归
linear_model = DecisionTreeRegressor(max_depth=8)

X_train = np.array(data_micro).reshape(-1, 1)
X_test = np.array(data_prediction).reshape(-1, 1)
y_train = data_nation

# 模型训练
linear_model.fit(X_train, y_train)
# 测试
y_pre = linear_model.predict(X_test)
for r in range(len(y_pre)):
    fit_output[r] = y_pre[r]

# Plot the results
plt.figure()
x = range(0, 1000)
plt.plot(x, y_train[:1000], color="b", label="实际值", linewidth=1.5)
plt.plot(x, y_pre[:1000], color="r", label="线性回归预测值", linewidth=1.5)  # (np.array(X_train)[:,i])[-1000:]

plt.xlabel("time/h")
plt.ylabel("value")
plt.title("prediction of PM25")
plt.legend()
plt.show()

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit\ANN_PM25_T_H_prediction.pare.fit.txt",
    'w')
for item in fit_output:
    for val in item:
        output_file.write('{:.4}\t'.format(val))
    output_file.write('\n')
output_file.close()

# 计算误差

