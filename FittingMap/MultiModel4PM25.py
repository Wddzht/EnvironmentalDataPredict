# coding=utf-8
# 只针对PM2.5, 对比实验，分几种方案：
# PM2.5+ T+ RH
# PM2.5+ T+ RH+ O3
# PM2.5+ T+ RH+ O3+CO;
# PM2.5+ T+ RH+ O3+CO+SO2

from pylab import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']  # TEMPERATURE	HUMIDITY _ _ _ _
N = 6
window = 3

# 读取数据
file_micro_train = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_micro.pare.txt")
data_micro_train = []
for line in file_micro_train:
    items = line.strip().split('\t')
    _list = []
    for item in items:
        _list.append(float(item))
    data_micro_train.append(_list)
data_micro_normal = np.zeros([len(data_micro_train), len(data_micro_train[0])], np.float)

# 归一化
max_data = (data_micro_train[0])[:]
min_data = (data_micro_train[0])[:]
for i in range(len(data_micro_train)):
    for j in range(len(data_micro_train[0])):
        if data_micro_train[i][j] > max_data[j]:
            max_data[j] = data_micro_train[i][j]
        elif data_micro_train[i][j] < min_data[j]:
            min_data[j] = data_micro_train[i][j]
for i in range(len(data_micro_train)):
    for j in range(len(data_micro_train[0])):
        data_micro_normal[i][j] = (data_micro_train[i][j] - min_data[j]) / (max_data[j] - min_data[j])

# Fit regression model
col = len(data_micro_train[0])

error_regr = np.zeros((N, 4), dtype=np.double)  # 4个误差函数
error_linear = np.zeros((N, 4), dtype=np.double)
error_rbf_svr = np.zeros((N, 4), dtype=np.double)
error_gbdt = np.zeros((N, 4), dtype=np.double)
error_rfr = np.zeros((N, 4), dtype=np.double)

# 线性回归
linear_model = LinearRegression()
# 决策树回归
decision_tree = DecisionTreeRegressor(max_depth=8)
# 径向基核函数配置支持向量机rbfSVR
rbf_svr = SVR(kernel="rbf")
# 随机森林
rfr = RandomForestRegressor()
# GBDT
gbdt = GradientBoostingRegressor(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)

train_data = np.array(data_micro_normal[:int(len(data_micro_normal) * 0.5)])
test_data = np.array(data_micro_normal)

linear_model_predict_data = np.zeros([len(test_data)])
decision_tree_predict_data = np.zeros([len(test_data)])
rbf_svr_predict_data = np.zeros([len(test_data)])
rfr_predict_data = np.zeros([len(test_data)])
gbdt_predict_data = np.zeros([len(test_data)])

# 只针对PM2.5 N=4
X_ = []
y_ = []
for i in range(len(data_micro_normal)):
    if i < window:
        list = []
        for j in range(window):
            #list.append(data_micro_normal[i + window + 1][0])  # CO
            #list.append(data_micro_normal[i + window + 1][3])  # O3
            list.append(data_micro_normal[i + window + 1][4])
            list.append(data_micro_normal[i + window + 1][6])  # T
            list.append(data_micro_normal[i + window + 1][7])  # H
        X_.append(list)
    else:
        list = []
        for j in range(window):
            #list.append(data_micro_normal[i - window - 1][0])
            #list.append(data_micro_normal[i - window - 1][3])
            list.append(data_micro_normal[i - window - 1][4])
            list.append(data_micro_normal[i - window - 1][6])
            list.append(data_micro_normal[i - window - 1][7])
        X_.append(list)
    y_.append(data_micro_normal[i][4])

X_train = X_[:int(len(data_micro_normal) * 0.7)]
X_test = X_
y_train = y_[:int(len(data_micro_normal) * 0.7)]
y_test = y_

# training
decision_tree.fit(X_train, y_train)
linear_model.fit(X_train, y_train)
rbf_svr.fit(X_train, y_train)
gbdt.fit(X_train, y_train)
rfr.fit(X_train, y_train)
joblib.dump(gbdt, 'train_gbdt.m')

# predict
decision_tree_y_predict = decision_tree.predict(X_test)
linear_regression_y_predict = linear_model.predict(X_test)
rbf_svr_y_predict = rbf_svr.predict(X_test)
gbdt_y_predict = gbdt.predict(X_test)
rfr_y_predict = rfr.predict(X_test)

# 反归一化
index_pm25 = 4
y_test_ = np.zeros((len(y_test)), dtype=np.double)
for k in range(len(y_test)):
    y_test_[k] = y_test[k] * (max_data[index_pm25] - min_data[index_pm25]) + min_data[index_pm25]
for k in range(len(decision_tree_y_predict)):
    decision_tree_y_predict[k] = decision_tree_y_predict[k] * (max_data[index_pm25] - min_data[index_pm25]) + min_data[
        index_pm25]
for k in range(len(linear_regression_y_predict)):
    linear_regression_y_predict[k] = linear_regression_y_predict[k] * (max_data[index_pm25] - min_data[index_pm25]) + \
                                     min_data[index_pm25]
for k in range(len(rbf_svr_y_predict)):
    rbf_svr_y_predict[k] = rbf_svr_y_predict[k] * (max_data[index_pm25] - min_data[index_pm25]) + min_data[index_pm25]
for k in range(len(gbdt_y_predict)):
    gbdt_y_predict[k] = gbdt_y_predict[k] * (max_data[index_pm25] - min_data[index_pm25]) + min_data[index_pm25]
for k in range(len(rfr_y_predict)):
    rfr_y_predict[k] = rfr_y_predict[k] * (max_data[index_pm25] - min_data[index_pm25]) + min_data[index_pm25]

# Plot the results
plt.figure()
x = range(1, len(test_data) + 1)
plt.plot(x, y_test_, color="b", label="ActualValue", linewidth=1.5)
plt.plot(x, linear_regression_y_predict, color="r", label="LinearRegression", linewidth=1.5)
plt.plot(x, decision_tree_y_predict, color="green", label="DecisionTreeRegressor", linewidth=1.5)
plt.plot(x, rbf_svr_y_predict, color="black", label="SVR", linewidth=1.5)
plt.plot(x, rfr_y_predict, color="g", label="RandomForestRegressor", linewidth=1.5)
plt.plot(x, gbdt_y_predict, color="y", label="GBDT", linewidth=1.5)

plt.xlabel("time")
plt.ylabel("Concentration(μg/m^3)")
plt.title("{:} at FuZhou,China".format(item_name[index_pm25]))
plt.legend()
plt.show()

print('{:.3f}\t{:.3f}\t{:.3f}'.format(
    decision_tree.score(X_test, y_test), mean_absolute_error(decision_tree_y_predict, y_test_),
    math.sqrt(mean_squared_error(decision_tree_y_predict, y_test_))
))
print('{:.3f}\t{:.3f}\t{:.3f}'.format(
    gbdt.score(X_test, y_test), mean_absolute_error(gbdt_y_predict, y_test_),
    math.sqrt(mean_squared_error(gbdt_y_predict, y_test_))
))
print('{:.3f}\t{:.3f}\t{:.3f}'.format(
    linear_model.score(X_test, y_test), mean_absolute_error(linear_regression_y_predict, y_test_),
    math.sqrt(mean_squared_error(linear_regression_y_predict, y_test_))
))
print('{:.3f}\t{:.3f}\t{:.3f}'.format(
    rfr.score(X_test, y_test), mean_absolute_error(rfr_y_predict, y_test_),
    math.sqrt(mean_squared_error(rfr_y_predict, y_test_))
))
print('{:.3f}\t{:.3f}\t{:.3f}'.format(
    rbf_svr.score(X_test, y_test), mean_absolute_error(rbf_svr_y_predict, y_test_),
    math.sqrt(mean_squared_error(rbf_svr_y_predict, y_test_))
))


# 输出预测值
# for t in range(len(y_test)):
#     linear_model_predict_data[t][i] = linear_regression_y_predict[t]
# for t in range(len(y_test)):
#     decision_tree_predict_data[t][i] = decision_tree_y_predict[t]
# for t in range(len(y_test)):
#     rbf_svr_predict_data[t][i] = rbf_svr_y_predict[t]
# for t in range(len(y_test)):
#     rfr_predict_data[t][i] = rfr_y_predict[t]
# for t in range(len(y_test)):
#     gbdt_predict_data[t][i] = gbdt_y_predict[t]

# error_regr[i, :] = [decision_tree.score(X_test, y_test), mean_squared_error(y_test_, decision_tree_y_predict),
#                     mean_absolute_error(y_test_, decision_tree_y_predict),
#                     math.sqrt(mean_squared_error(y_test_, decision_tree_y_predict))]
# error_linear[i, :] = [linear_model.score(X_test, y_test), mean_squared_error(y_test_, linear_regression_y_predict),
#                       mean_absolute_error(y_test_, linear_regression_y_predict),
#                       math.sqrt(mean_squared_error(y_test_, linear_regression_y_predict))]
# error_rbf_svr[i, :] = [rbf_svr.score(X_test, y_test), mean_squared_error(y_test_, rbf_svr_y_predict),
#                        mean_absolute_error(y_test_, rbf_svr_y_predict),
#                        math.sqrt(mean_squared_error(y_test_, rbf_svr_y_predict))]
# error_rfr[i, :] = [rfr.score(X_test, y_test), mean_squared_error(y_test_, rfr_y_predict),
#                    mean_absolute_error(y_test_, rfr_y_predict),
#                    math.sqrt(mean_squared_error(y_test_, rfr_y_predict))]
# error_gbdt[i, :] = [gbdt.score(X_test, y_test), mean_squared_error(y_test_, gbdt_y_predict),
#                     mean_absolute_error(y_test_, gbdt_y_predict),
#                     math.sqrt(mean_squared_error(y_test_, gbdt_y_predict))]
