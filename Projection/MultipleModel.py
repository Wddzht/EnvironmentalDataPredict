# coding=utf-8
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import pymc3 as pm
import matplotlib.pyplot as plt
import xlrd
import xlwt
import math

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# Decision Tree Regression Portland and Fuzhou2017010618
# 读取数据，划分训练集和测试集
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 6  # 污染物的个数
tr = 0.7
data = xlrd.open_workbook(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\Fuzhou160032_700.xlsx')
# data = xlrd.open_workbook('2Portland2019010618.xlsx')
# 读取数据
table = data.sheets()[0]  # 通过索引顺序获取
row = table.nrows  # 数据表的行数
col = table.ncols

data_norm = np.zeros((table.nrows-1, table.ncols), dtype=np.double)
max_data = table.row_values(1)
min_data = table.row_values(1)
# 归一化
for i in range(1, row):
    rows = table.row_values(i)
    for j in range(0, col):
        if rows[j] > max_data[j]:
            max_data[j] = rows[j]
        elif rows[j] < min_data[j]:
            min_data[j] = rows[j]
for i in range(1, row):
    rows = table.row_values(i)
    for j in range(0, col):
        data_norm[i - 1][j] = (rows[j] - min_data[j]) / (max_data[j] - min_data[j])

tr_length = math.ceil((row - 1) * tr)
te_length = row - 1 - tr_length
train_data = np.zeros((tr_length, col), dtype=np.double)  # 训练集
test_data = np.zeros(((row - 1) - tr_length, col), dtype=np.double)  # 测试集

# for i in range(1, row):
#     rows = np.matrix(table.row_values(i))  # 把list转换为矩阵进行矩阵操作
#     rows[rows == ''] = float(0)
#     if i <= tr_length:
#         train_data[i - 1, :] = rows  # 按行把数据存进矩阵中
#     else:
#         test_data[i - 1 - tr_length, :] = rows
for i in range(row-1):
    if i < tr_length:
        train_data[i] = data_norm[i]  # 按行把数据存进矩阵中
    else:
        test_data[i - tr_length] = data_norm[i]

# Fit regression model
a = np.zeros((tr_length, col - 1), dtype=np.double)
b = np.zeros(((row - 1) - tr_length, col - 1), dtype=np.double)

error_regr = np.zeros((N, 4), dtype=np.double)  # 4个误差函数
error_linear = np.zeros((N, 4), dtype=np.double)
error_rbf_svr = np.zeros((N, 4), dtype=np.double)
error_gbdt = np.zeros((N, 4), dtype=np.double)
error_rfr = np.zeros((N, 4), dtype=np.double)

# 线性回归
linear_model = LinearRegression()
# 决策树回归
regr = DecisionTreeRegressor(max_depth=8)
# 线性核函数配置支持向量机linearSVR
linear_svr = SVR(kernel="linear")
# 多项式核函数配置支持向量机polySVR
poly_svr = SVR(kernel="poly")
# 径向基核函数配置支持向量机rbfSVR
rbf_svr = SVR(kernel="rbf")
# 随机森林
rfr = RandomForestRegressor()
# GBDT
gbdt = GradientBoostingRegressor(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)

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

    # training
    regr.fit(X_train, y_train)
    linear_model.fit(X_train, y_train)
    # linear_svr.fit(X_train, y_train)
    # poly_svr.fit(X_train, y_train)
    rbf_svr.fit(X_train, y_train)
    gbdt.fit(X_train, y_train)
    rfr.fit(X_train, y_train)
    joblib.dump(gbdt, 'train_gbdt.m')  # 保存模型

    # predict
    X_test = b
    y_test = b1
    decision_tree_y_predict = regr.predict(X_test)
    linear_regression_y_predict = linear_model.predict(X_test)
    # linear_svr_y_predict = linear_svr.predict(X_test)
    # poly_svr_y_predict = poly_svr.predict(X_test)
    rbf_svr_y_predict = rbf_svr.predict(X_test)
    gbdt_y_predict = gbdt.predict(X_test)
    rfr_y_predict = rfr.predict(X_test)

    # 反归一化
    y_test_=np.zeros((len(y_test)), dtype=np.double)
    for k in range(len(y_test)):
        y_test_[k] = y_test[k] * (max_data[i] - min_data[i]) + min_data[i]
    for k in range(len(decision_tree_y_predict)):
        decision_tree_y_predict[k] = decision_tree_y_predict[k] * (max_data[i] - min_data[i]) + min_data[i]
    for k in range(len(linear_regression_y_predict)):
        linear_regression_y_predict[k] = linear_regression_y_predict[k] * (max_data[i] - min_data[i]) + min_data[i]
    for k in range(len(rbf_svr_y_predict)):
        rbf_svr_y_predict[k] = rbf_svr_y_predict[k] * (max_data[i] - min_data[i]) + min_data[i]
    for k in range(len(gbdt_y_predict)):
        gbdt_y_predict[k] = gbdt_y_predict[k] * (max_data[i] - min_data[i]) + min_data[i]
    for k in range(len(rfr_y_predict)):
        rfr_y_predict[k] = rfr_y_predict[k] * (max_data[i] - min_data[i]) + min_data[i]

    # Plot the results
    plt.figure()
    x = range(1, te_length + 1)
    plt.plot(x, y_test_, color="b", label="实际值", linewidth=1.5)
    plt.plot(x, linear_regression_y_predict, color="r", label="线性回归预测值", linewidth=1.5)
    plt.plot(x, decision_tree_y_predict, color="green", label="决策树预测", linewidth=1.5)
    # plt.plot(x, linear_svr_y_predict, color="green", label="linearSVR", linewidth=1.5)
    # plt.plot(x, poly_svr_y_predict, color="pink", label="polySVR", linewidth=1.5)
    plt.plot(x, rbf_svr_y_predict, color="black", label="rbfSVR", linewidth=1.5)
    plt.plot(x, rfr_y_predict, color="g", label="随机森林", linewidth=1.5)
    plt.plot(x, gbdt_y_predict, color="y", label="Gradient Boosting Decision Tree", linewidth=1.5)

    plt.xlabel("time/h")
    plt.xlabel("time/h")
    plt.ylabel("value")
    plt.title("数据预测-" + item_name[i])
    plt.legend()
    plt.show()

    output_file = open(
        "E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\\muti_model\\" + item_name[i] + ".txt", 'w')
    output_file.write("实际值\t线性回归\t决策树\tSVR\t随机深林\tGBDT\n")
    for t in range(len(y_test)):
        output_file.write(
            str(y_test_[t]) + '\t' + str(linear_regression_y_predict[t]) + '\t' + str(decision_tree_y_predict[t])
            + '\t' + str(rbf_svr_y_predict[t]) + '\t' + str(rfr_y_predict[t]) + '\t' + str(gbdt_y_predict[t]) + '\n')
    output_file.close()

    error_regr[i, :] = [regr.score(X_test, y_test), mean_squared_error(y_test_, decision_tree_y_predict),
                        mean_absolute_error(y_test_, decision_tree_y_predict),
                        math.sqrt(mean_squared_error(y_test_, decision_tree_y_predict))]
    error_linear[i, :] = [linear_model.score(X_test, y_test), mean_squared_error(y_test_, linear_regression_y_predict),
                          mean_absolute_error(y_test_, linear_regression_y_predict),
                          math.sqrt(mean_squared_error(y_test_, linear_regression_y_predict))]
    error_rbf_svr[i, :] = [rbf_svr.score(X_test, y_test), mean_squared_error(y_test_, rbf_svr_y_predict),
                           mean_absolute_error(y_test_, rbf_svr_y_predict),
                           math.sqrt(mean_squared_error(y_test_, rbf_svr_y_predict))]
    error_rfr[i, :] = [rfr.score(X_test, y_test), mean_squared_error(y_test_, rfr_y_predict),
                       mean_absolute_error(y_test_, rfr_y_predict),
                       math.sqrt(mean_squared_error(y_test_, rfr_y_predict))]
    error_gbdt[i, :] = [gbdt.score(X_test, y_test), mean_squared_error(y_test_, gbdt_y_predict),
                        mean_absolute_error(y_test_, gbdt_y_predict),
                        math.sqrt(mean_squared_error(y_test_, gbdt_y_predict))]

output_rate = open(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\Projection\rate_muti.txt', 'w')

output_rate.write('决策树\n')
for r in error_regr:
    for j in range(len(r)):
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')

output_rate.write('线性回归\n')
for r in error_linear:
    for j in range(len(r)):
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')

output_rate.write('SVR(rbf)\n')
for r in error_rbf_svr:
    for j in range(len(r)):
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')

output_rate.write('随机深林\n')
for r in error_rfr:
    for j in range(len(r)):
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')

output_rate.write('GBDT\n')
for r in error_gbdt:
    for j in range(len(r)):
        output_rate.write(str(r[j]) + '\t')
    output_rate.write('\n')

output_rate.close()
