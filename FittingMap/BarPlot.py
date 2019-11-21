from pylab import *
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime
import numpy as np
import os

color = ['hotpink', 'yellow', 'dimgray', 'g', 'b', 'orange']
linestyle = ['-', '-', '-', '-', '-', '-', '-']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
molecular_weight = {'CO': 28, 'NO2': 46, 'SO2': 64, 'O3': 48}  # 分子量
window_len = 6
attr_count = 6

nation_file_fz = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_path_fz = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit\_小输入预测\GBDT.pare.fit.txt'
fig_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\bar_plot\bar.png'

nation_file_ptl = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
file_path_ptl = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit\_小输入预测\GBDT.pare.fit.txt'

figsize = (9.5, 4)
markersize = 4
fontsize = 11

range_dict = {
    'CO': [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.75, 1), (1, 1.25), (1.25, 1.5), (1.5, 1.75),
           (1.75, 2.5)],
    'NO2': [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 100), (100, 150)],
    'O3': [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)],
    'PM25': [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 65)]}

# range_dict = {'CO': [(0, 10)],
#               'NO2': [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 100), (100, 150)],
#               'O3': [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 250)],
#               'PM25': [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 80)]}

bar_count = len(range_dict['CO']) + len(range_dict['NO2']) + len(range_dict['O3']) + len(range_dict['PM25'])


def cal_err(actual_valuie, prediction_value):
    return mean_absolute_error(actual_valuie, prediction_value)


# 国测站数据
nation_data_fz = []
for line in nation_file_fz:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    nation_data_fz.append(_list)

nation_data_ptl = []
for line in nation_file_ptl:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    nation_data_ptl.append(_list)

fit_file_fz = open(file_path_fz)
fit_data_fz = []
for line in fit_file_fz:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    fit_data_fz.append(_list)

fit_file_ptl = open(file_path_ptl)
fit_data_ptl = []
for line in fit_file_ptl:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    fit_data_ptl.append(_list)

bias_of_attr = []
plt.figure(figsize=figsize)
count_index = 0
bar_width = 0.3

for attr in range(attr_count):
    err_list_ptl = []
    err_list_fz = []

    if item_name[attr] in ['SO2', 'PM10']:
        continue
    real_value_fz = np.array(nation_data_fz)[:, attr]
    real_value_ptl = np.array(nation_data_ptl)[:, attr]
    fit_attr_fz = np.array(fit_data_fz)[:, attr]
    fit_attr_ptl = np.array(fit_data_ptl)[:, attr]
    # 国测站测量值 单位换算 （只有福州数据需要）
    if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
        real_value_fz = (molecular_weight[item_name[attr]] / 22.4) * real_value_fz
    if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
        fit_attr_fz = (molecular_weight[item_name[attr]] / 22.4) * fit_attr_fz

    plt.subplots_adjust(bottom=0.22, top=0.9)
    # plt.ylabel("MAE (GBDT)", fontsize=fontsize)

    # 按浓度分组
    # if item_name[attr] == 'CO':
    for ran_b, ran_t in range_dict[item_name[attr]]:
        temp_fz = [[], []]
        for target, fit in zip(real_value_fz[:len(fit_attr_fz)], fit_attr_fz):
            if ran_b <= target < ran_t:
                temp_fz[0].append(target)
                temp_fz[1].append(fit)
        if len(temp_fz[0]) > 0:
            # plt.plot([xi for xi in range(len(temp_fz[0]))], temp_fz[0], label=item_name[attr] + 'actual value')
            # plt.plot([xi for xi in range(len(temp_fz[0]))], temp_fz[1], label=item_name[attr] + 'prediction')
            # plt.legend(fontsize=fontsize)
            # plt.show()

            err_list_fz.append(cal_err(temp_fz[0], temp_fz[1]))
        else:
            err_list_fz.append(0)
        temp_ptl = [[], []]
        for target, fit in zip(real_value_ptl[:len(fit_attr_ptl)], fit_attr_ptl):
            if ran_b <= target < ran_t:
                temp_ptl[0].append(target)
                temp_ptl[1].append(fit)
        if len(temp_ptl[0]) > 0:
            err_list_ptl.append(cal_err(temp_ptl[0], temp_ptl[1]))
        else:
            err_list_ptl.append(0)

    plt.subplot(1, 4, 1 + count_index)
    length = len(range_dict[item_name[attr]])
    plt.bar(np.arange(length), err_list_ptl, width=bar_width, color='b')
    plt.bar(np.arange(length) + bar_width, err_list_fz, width=bar_width, color='r')
    plt.xticks(np.arange(0, length, 2),
               [range_dict[item_name[attr]][i][0] for i in range(length) if i % 2 == 0])
    count_index += 1
plt.show()
