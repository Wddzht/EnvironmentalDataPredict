# coding=utf-8
from pylab import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']  # _6(T)  _7(H)
inform_matric = {'CO': [0, 6],
                 'NO2': [1, 7],
                 'SO2': [2],
                 'O3': [1, 3, 6, 7],
                 'PM25': [4,3],
                 'PM10': [3, 5, 6, 7]}  # 参数矩阵
N = 6
window = 1
is_output_data = False

output_path = r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction\\"
# 读取数据
file_micro_train = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt")
data_micro_train = []
for line in file_micro_train:
    items = line.strip().split('\t')
    _list = []
    for item in items:
        _list.append(float(item))
    data_micro_train.append(_list)

data_micro_normal = np.zeros_like(data_micro_train, np.float)
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
        if max_data[j] - min_data[j] == 0:
            data_micro_normal[i][j] = data_micro_train[i][j] - min_data[j]
        else:
            data_micro_normal[i][j] = (data_micro_train[i][j] - min_data[j]) / (max_data[j] - min_data[j])

# Fit regression model
col = len(data_micro_train[0])

error_regr = np.zeros((N, 3), dtype=np.double)  # 3个误差函数
error_linear = np.zeros((N, 3), dtype=np.double)
error_rbf_svr = np.zeros((N, 3), dtype=np.double)
error_gbdt = np.zeros((N, 3), dtype=np.double)
error_rfr = np.zeros((N, 3), dtype=np.double)

# 线性回归 决策树回归 径向基核函数配置支持向量机rbfSVR 随机森林 GBDT
linear_model = LinearRegression()
decision_tree = DecisionTreeRegressor(max_depth=8)
rbf_svr = SVR(kernel="rbf")
rfr = RandomForestRegressor(n_estimators=100)
gbdt = GradientBoostingRegressor(n_estimators=100, max_depth=8, min_samples_split=2, learning_rate=0.1)

linear_model_predict_data = np.zeros_like(data_micro_normal)
decision_tree_predict_data = np.zeros_like(data_micro_normal)
rbf_svr_predict_data = np.zeros_like(data_micro_normal)
rfr_predict_data = np.zeros_like(data_micro_normal)
gbdt_predict_data = np.zeros_like(data_micro_normal)

for i in range(0, N):
    X_ = []
    y_ = []
    for index in range(len(data_micro_normal)):
        if index < window:
            _list = []
            for j in range(window):
                for k in inform_matric[item_name[i]]:
                    _list.append(data_micro_normal[index + window + 1][k])
            X_.append(_list)
        else:
            _list = []
            for j in range(window):
                for k in inform_matric[item_name[i]]:
                    _list.append(data_micro_normal[index - window - 1][k])
            X_.append(_list)
        y_.append(data_micro_normal[index][i])

    X_train = X_[:int(len(data_micro_normal) * 0.7)]
    X_test = X_[int(len(data_micro_normal) * 0.6):]
    y_train = y_[:int(len(data_micro_normal) * 0.7)]
    y_test = y_[int(len(data_micro_normal) * 0.6):]

    # training
    decision_tree.fit(X_train, y_train)
    linear_model.fit(X_train, y_train)
    rbf_svr.fit(X_train, y_train)
    gbdt.fit(X_train, y_train)
    rfr.fit(X_train, y_train)
    # joblib.dump(gbdt, 'train_gbdt.m')

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
    x = range(1, len(y_test_) + 1)
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
    # plt.show()

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

    error_regr[i, :] = [r2_score(y_test_, decision_tree_y_predict),
                        mean_absolute_error(y_test_, decision_tree_y_predict),
                        math.sqrt(mean_squared_error(y_test_, decision_tree_y_predict))]
    error_linear[i, :] = [r2_score(y_test_, linear_regression_y_predict),
                          mean_absolute_error(y_test_, linear_regression_y_predict),
                          math.sqrt(mean_squared_error(y_test_, linear_regression_y_predict))]
    error_rbf_svr[i, :] = [r2_score(y_test_, rbf_svr_y_predict), mean_absolute_error(y_test_, rbf_svr_y_predict),
                           math.sqrt(mean_squared_error(y_test_, rbf_svr_y_predict))]
    error_rfr[i, :] = [r2_score(y_test_, rfr_y_predict), mean_absolute_error(y_test_, rfr_y_predict),
                       math.sqrt(mean_squared_error(y_test_, rfr_y_predict))]
    error_gbdt[i, :] = [r2_score(y_test_, gbdt_y_predict), mean_absolute_error(y_test_, gbdt_y_predict),
                        math.sqrt(mean_squared_error(y_test_, gbdt_y_predict))]

if is_output_data:
    output_file = open(output_path + "LinearRegression.pare.txt", 'w')
    for data in linear_model_predict_data:
        output_file.write('\t'.join(list(map(str, data))) + '\n')
    output_file.close()

    output_file = open(output_path + "DecisionTree.pare.txt", 'w')
    for data in decision_tree_predict_data:
        output_file.write('\t'.join(list(map(str, data))) + '\n')
    output_file.close()

    output_file = open(output_path + "SVR.pare.txt", 'w')
    for data in rbf_svr_predict_data:
        output_file.write('\t'.join(list(map(str, data))) + '\n')
    output_file.close()

    output_file = open(output_path + "RFR.pare.txt", 'w')
    for data in rfr_predict_data:
        output_file.write('\t'.join(list(map(str, data))) + '\n')
    output_file.close()

    output_file = open(output_path + "GBDT.pare.txt", 'w')
    for data in gbdt_predict_data:
        output_file.write('\t'.join(list(map(str, data))) + '\n')
    output_file.close()

output_rate = open(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\output_rate.txt', 'w')
output_rate.write('污染物\tR2\tMAE\tRMSE\n')
for i in range(N):
    output_rate.write(item_name[i] + '\n')
    output_rate.write('{:.2f}\t'.format(error_regr[i][0]))
    output_rate.write('{:.2f}\t'.format(error_regr[i][1]))
    output_rate.write('{:.2f}\t\n'.format(error_regr[i][2]))
    output_rate.write('{:.2f}\t'.format(error_gbdt[i][0]))
    output_rate.write('{:.2f}\t'.format(error_gbdt[i][1]))
    output_rate.write('{:.2f}\t\n'.format(error_gbdt[i][2]))
    output_rate.write('{:.2f}\t'.format(error_linear[i][0]))
    output_rate.write('{:.2f}\t'.format(error_linear[i][1]))
    output_rate.write('{:.2f}\t\n'.format(error_linear[i][2]))
    output_rate.write('{:.2f}\t'.format(error_rfr[i][0]))
    output_rate.write('{:.2f}\t'.format(error_rfr[i][1]))
    output_rate.write('{:.2f}\t\n'.format(error_rfr[i][2]))
    output_rate.write('{:.2f}\t'.format(error_rbf_svr[i][0]))
    output_rate.write('{:.2f}\t'.format(error_rbf_svr[i][1]))
    output_rate.write('{:.2f}\t\n'.format(error_rbf_svr[i][2]))
output_rate.close()
