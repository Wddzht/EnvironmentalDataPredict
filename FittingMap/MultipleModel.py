# coding=utf-8
from pylab import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
N = 6

# 读取数据
file_micro_train = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt")
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


train_data = np.array(data_micro_normal[:int(len(data_micro_normal) * 0.7)])
test_data = np.array(data_micro_normal)

linear_model_predict_data = np.zeros([len(test_data), len(test_data[0])])
decision_tree_predict_data = np.zeros([len(test_data), len(test_data[0])])
rbf_svr_predict_data = np.zeros([len(test_data), len(test_data[0])])
rfr_predict_data = np.zeros([len(test_data), len(test_data[0])])
gbdt_predict_data = np.zeros([len(test_data), len(test_data[0])])

for i in range(0, N):
    y_train = train_data[:, i]
    y_test = test_data[:, i]
    k = 0
    X_train = [[0 for m in range(col - 1)] for j in range(len(train_data))]
    X_test = [[0 for m in range(col - 1)] for j in range(len(test_data))]
    for j in range(0, col):
        if j != i:
            a2 = train_data[:, j]
            b2 = test_data[:, j]
            for t in range(len(a2)):
                X_train[t][k] = a2[t]
            for t in range(len(b2)):
                X_test[t][k] = b2[t]
            k += 1

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
    y_test_ = np.zeros((len(y_test)), dtype=np.double)
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
    x = range(1, len(test_data) + 1)
    plt.plot(x, y_test_, color="b", label="实际值", linewidth=1.5)
    plt.plot(x, linear_regression_y_predict, color="r", label="线性回归预测值", linewidth=1.5)
    plt.plot(x, decision_tree_y_predict, color="green", label="决策树预测", linewidth=1.5)
    plt.plot(x, rbf_svr_y_predict, color="black", label="rbfSVR", linewidth=1.5)
    plt.plot(x, rfr_y_predict, color="g", label="随机森林", linewidth=1.5)
    plt.plot(x, gbdt_y_predict, color="y", label="Gradient Boosting Decision Tree", linewidth=1.5)

    plt.xlabel("time/h")
    plt.ylabel("value")
    plt.title("数据预测-" + item_name[i])
    plt.legend()
    plt.show()

    # 输出预测值
    for t in range(len(y_test)):
        linear_model_predict_data[t][i] = linear_regression_y_predict[t]
    for t in range(len(y_test)):
        decision_tree_predict_data[t][i] = decision_tree_y_predict[t]
    for t in range(len(y_test)):
        rbf_svr_predict_data[t][i] = rbf_svr_y_predict[t]
    for t in range(len(y_test)):
        rfr_predict_data[t][i] = rfr_y_predict[t]
    for t in range(len(y_test)):
        gbdt_predict_data[t][i] = gbdt_y_predict[t]

    error_regr[i, :] = [decision_tree.score(X_test, y_test), mean_squared_error(y_test_, decision_tree_y_predict),
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

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\LM_prediction.pare.txt", 'w')
for data in linear_model_predict_data:
    output_file.write('\t'.join(list(map(str,data)))+'\n')
output_file.close()

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\Decision_tree_prediction.pare.txt", 'w')
for data in decision_tree_predict_data:
    output_file.write('\t'.join(list(map(str,data)))+'\n')
output_file.close()

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\SVR.pare.txt", 'w')
for data in rbf_svr_predict_data:
    output_file.write('\t'.join(list(map(str,data)))+'\n')
output_file.close()

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\RFR_prediction.pare.txt", 'w')
for data in rfr_predict_data:
    output_file.write('\t'.join(list(map(str,data)))+'\n')
output_file.close()

output_file = open(
    r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\GBDT_prediction.pare.txt", 'w')
for data in gbdt_predict_data:
    output_file.write('\t'.join(list(map(str,data)))+'\n')
output_file.close()

# output_rate = open(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\output_rate.txt', 'w')
# output_rate.write('决策树\n')
# for r in error_regr:
#     for j in range(len(r)):
#         output_rate.write(str(r[j]) + '\t')
#     output_rate.write('\n')
#
# output_rate.write('线性回归\n')
# for r in error_linear:
#     for j in range(len(r)):
#         output_rate.write(str(r[j]) + '\t')
#     output_rate.write('\n')
#
# output_rate.write('SVR(rbf)\n')
# for r in error_rbf_svr:
#     for j in range(len(r)):
#         output_rate.write(str(r[j]) + '\t')
#     output_rate.write('\n')
#
# output_rate.write('随机深林\n')
# for r in error_rfr:
#     for j in range(len(r)):
#         output_rate.write(str(r[j]) + '\t')
#     output_rate.write('\n')
#
# output_rate.write('GBDT\n')
# for r in error_gbdt:
#     for j in range(len(r)):
#         output_rate.write(str(r[j]) + '\t')
#     output_rate.write('\n')
#
# output_rate.close()
