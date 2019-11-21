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

# 单位换算 （只有福州数据需要）
# value_name = 'FuZhou,China'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit\_小输入预测'
fig_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_plot\{:}.png'

# value_name = 'Portland,USA'
# nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
# file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit\_小输入预测'
# fig_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_plot\{:}.png'

figsize = (9.5, 4)
markersize = 4
fontsize = 11

# 国测站数据
nation_data = []
for line in nation_file:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    nation_data.append(_list)

bias_of_attr = []
filenames = os.listdir(file_path)
plt.figure(figsize=figsize)
count_index = 0
bp_list = []

for attr in range(attr_count):
    if item_name[attr] in ['SO2', 'PM10']:
        continue
    real_value = np.array(nation_data)[:, attr]
    # 国测站测量值 单位换算 （只有福州数据需要）
    if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
        real_value = (molecular_weight[item_name[attr]] / 22.4) * real_value

    plt.subplots_adjust(bottom=0.22, top=0.9)

    # 单位修改，国内CO:mg/m3，SO2/N O2/O3/颗粒物是ug/m3
    # 波特兰国测站气体单位CO:ppm，SO2/N O2/O3:ppb，颗粒物是ug/m3。
    # mg / m3 = M / 22.4 * ppm，M为气体的分子量
    if item_name[attr] in ['CO']:
        plt.ylabel("Concentration(ppm)", fontsize=fontsize)
    if item_name[attr] in ['SO2', 'NO2', 'O3']:
        plt.ylabel("Concentration(ppb)", fontsize=fontsize)
    if item_name[attr] in ['PM25', 'PM10']:
        plt.ylabel("Concentration(μg/m^3)", fontsize=fontsize)

    color_index = 0
    model_name = []
    for filename in filenames:
        if filename[0] == '_':
            continue
        fit_file = open(os.path.join(file_path, filename))
        fit_data = []
        if filename[:3] == 'ANN':
            for i in range(window_len):
                fit_data.append([0, 0, 0, 0, 0, 0])  # 对齐窗口

        for line in fit_file:
            line = line.strip()
            if not line:
                continue
            _list = []
            for i in line.split('\t'):
                _list.append(float(i))
            fit_data.append(_list)
        fit_data = np.array(fit_data)[:, attr]  # todo:

        # 预测值单位换算（只有福州数据需要）
        if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
            fit_data = (molecular_weight[item_name[attr]] / 22.4) * fit_data
        bias_of_attr.append(fit_data - real_value[:len(fit_data)])
        model_name.append(filename.split('.')[0])

        bp_list.append(plt.boxplot(fit_data - real_value[:len(fit_data)],
                                   showfliers=False,
                                   positions=[color_index + (count_index) * 5],  # TODO:5个模型一个循环
                                   boxprops={'facecolor': color[color_index]},
                                   medianprops={'color': 'k'},
                                   patch_artist=True))
        color_index += 1
    count_index += 1

    # 讨论不同湿度对预测影响



    # plt.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0],
    #             bp_list[2]["boxes"][0], bp_list[3]["boxes"][0],
    #             bp_list[4]["boxes"][0]], model_name, loc='upper right')
    # plt.show()

# plt.legend([bp_list[0]["boxes"][0], bp_list[1]["boxes"][0],
#             bp_list[2]["boxes"][0], bp_list[3]["boxes"][0],
#             bp_list[4]["boxes"][0], bp_list[5]["boxes"][0]], model_name, loc='upper right')
# plt.show()
