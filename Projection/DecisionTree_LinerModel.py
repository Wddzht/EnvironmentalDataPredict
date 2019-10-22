# coding=utf-8
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import xlrd
import xlwt

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# Decision Tree Regression Portland and Fuzhou2017010618
# 读取数据，划分训练集和测试集
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 6  # 污染物的个数
tr = 0.7
data = xlrd.open_workbook(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\3Fuzhou2017010618.xlsx')
# data = xlrd.open_workbook('2Portland2019010618.xlsx')
# 读取数据
table = data.sheets()[0]  # 通过索引顺序获取
row = table.nrows  # 数据表的行数
col = table.ncols

tr_length = math.ceil((row - 1) * tr)
te_length = row - 1 - tr_length
train_data = np.zeros((tr_length, col), dtype=np.double)  # 训练集
test_data = np.zeros(((row - 1) - tr_length, col), dtype=np.double)  # 测试集

for i in range(1, row):
    rows = np.matrix(table.row_values(i))  # 把list转换为矩阵进行矩阵操作
    rows[rows == ''] = float(0)
    if i <= tr_length:
        train_data[i - 1, :] = rows  # 按行把数据存进矩阵中
    else:
        test_data[i - 1 - tr_length, :] = rows

# Fit regression model
a = np.zeros((tr_length, col - 1), dtype=np.double)
b = np.zeros(((row - 1) - tr_length, col - 1), dtype=np.double)

error = np.zeros((N, 4), dtype=np.double)  # 4个误差函数
error_linear = np.zeros((N, 4), dtype=np.double)  # 4个误差函数
# 决策树回归
linear_model = LinearRegression()
# 多元线性回归
regr = DecisionTreeRegressor(max_depth=8)
for i in range(0, N):
    a1 = train_data[:, i]
    b1 = test_data[:, i]
    k = 0
    a = [[0 for m in range(col - 1)] for j in range(len(train_data))]
    b = [[0 for m in range(col - 1)] for j in range(len(test_data))]
    for j in range(0, col):
        if j != i:
            a2 = train_data[:, j]
            b2 = test_data[:, j]
            for t in range(len(a2)):
                a[t][k] = a2[t]
            for t in range(len(b2)):
                b[t][k] = b2[t]
            k += 1
    X_train = a
    y_train = a1
    # 模型
    regr.fit(X_train, y_train)
    linear_model.fit(X_train, y_train)
    # predict

    # decision tree
    X_test = b
    y_test = b1
    y_pre = regr.predict(X_test)

    # linear regression
    y_pre2 = linear_model.predict(X_test)

    # Plot the results
    plt.figure()
    x = range(1, te_length + 1)
    plt.plot(x, y_test, color="b", label="实际值", linewidth=1.5)

    plt.plot(x, y_pre2, color="r", label="线性回归预测值", linewidth=1.5)

    output_file = open("E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\\" + item_name[i] + ".txt", 'w')
    for t in range(len(y_test)):
        output_file.write(str(y_test[t]) + '\t' + str(y_pre2[t]) + '\t' + str(y_pre[t]) + '\n')
    output_file.close()

    plt.plot(x, y_pre, color="green", label="决策树预测", linewidth=1.5)
    plt.xlabel("time/h")
    plt.ylabel("value")
    plt.title("数据预测-" + item_name[i])
    plt.legend()
    plt.show()
    error[i, :] = [regr.score(X_test, y_test), mean_squared_error(y_test, y_pre), mean_absolute_error(y_test, y_pre),
                   math.sqrt(mean_squared_error(y_test, y_pre))]
    error_linear[i, :] = [linear_model.score(X_test, y_test), mean_squared_error(y_test, y_pre2),
                          mean_absolute_error(y_test, y_pre2),
                          math.sqrt(mean_squared_error(y_test, y_pre2))]
# print(error)

f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

# 将数据写入第 i 行，第 j 列

output_rate = open(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\rate.txt', 'w')
i = 0
output_rate.write('DecisionTreeRegressor\n')
for r in error:
    for j in range(len(r)):
        sheet1.write(i, j, r[j])
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')
    i = i + 1
output_rate.write('Linear Model\n')
for r in error_linear:
    for j in range(len(r)):
        sheet1.write(i, j, r[j])
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')
    i = i + 1

output_rate.close()

f.save(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\error_Fuzhou.xlsx')
# f.save('E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\error_Portland.xlsx')
