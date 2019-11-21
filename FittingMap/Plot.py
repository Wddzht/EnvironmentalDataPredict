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
plot_len = 600

# 单位换算 （只有福州数据需要）
value_name = '{:} at FuZhou,China'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit\_小输入预测'
fig_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_plot\{:}.png'

# value_name = '{:} at Portland,USA'
# nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
# file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit\_小输入预测'
# fig_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_plot\{:}.png'

plt_start_time = '17/02/01'  # 福州数据绘制起始时间
t = datetime.datetime.strptime('2017/1/8 18:00', '%Y/%m/%d %H:%M')  # 福州数据起始时间

# t = datetime.datetime.strptime('2018/12/19 0:00', '%Y/%m/%d %H:%M')  # 波特兰数据起始时间
# plt_start_time = '19/02/01'  # 波特兰数据绘制起始时间

figsize = (9.5, 4)
markersize = 4
fontsize = 11


def calculate_err(real_value, fit_value):
    r2 = r2_score(real_value, fit_value)
    # mae = mean_absolute_error(real_value, fit_value)
    # rmse = math.sqrt(mean_squared_error(real_value, fit_value))
    mnb = 0
    count = 1
    for o, m in zip(real_value, fit_value):
        if o == 0:
            continue
        mnb = mnb + (m - o) / o
        count += 1
    mnb = mnb / count

    mne = 0
    count = 1
    for o, m in zip(real_value, fit_value):
        if o == 0:
            continue
        mne = mne + abs(m - o) / o
        count += 1
    mne = mne / count
    return r2, mnb, mne


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

# x轴年月日序列
x_datetime = []
for i in range(len(nation_data)):
    x_datetime.append(t.strftime('%y/%m/%d'))
    t = t + datetime.timedelta(hours=1)

plot_start = x_datetime.index(plt_start_time)  # 数据绘制起始时间
x_index = [xi for xi in range(len(x_datetime) + 1) if xi % 48 == 0]
x_datetime = [x_datetime[xi] for xi in range(len(x_datetime) + 1) if xi % 48 == 0]

filenames = os.listdir(file_path)
for attr in range(attr_count):
    real_value = np.array(nation_data)[:, attr]
    # 国测站测量值 单位换算 （只有福州数据需要）
    if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
        real_value = (molecular_weight[item_name[attr]] / 22.4) * real_value

    plt.figure(figsize=figsize)
    plt.subplots_adjust(bottom=0.22, top=0.9)
    # plt.xlabel("Time", fontsize=9.5)
    plt.xticks(x_index, x_datetime, fontsize=9.5, rotation=30)

    plt.title(value_name.format(item_name[attr]), fontsize=12)

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
        fit_data = (np.array(fit_data)[:, attr])  # todo:

        # 预测值单位换算（只有福州数据需要）
        if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
            fit_data = (molecular_weight[item_name[attr]] / 22.4) * fit_data

        plt.plot([xi for xi in range(plot_start, plot_start + plot_len)],
                 fit_data[plot_start:plot_start + plot_len],
                 color=color[color_index],
                 label=filename.split('.')[0],
                 linewidth=1,
                 linestyle=linestyle[color_index])
        plt.plot([xi for xi in range(plot_start, plot_start + plot_len)], fit_data[plot_start:plot_start + plot_len],
                 '.', markersize=markersize,
                 color=color[color_index])

        color_index += 1

        r2, mnb, mne = calculate_err(real_value[window_len:len(fit_data)], fit_data[window_len:])
        # print('{:}-{:}\n{:.1f}\t{:.1f}\t{:.1f}\t\n'.format(filename.split('.')[0], item_name[attr], r2, mnb, mne))

        print('{:.1f}\t{:.1f}\t{:.1f}\t'.format(r2, mnb * 100, mne * 100))
    plt.plot([xi for xi in range(plot_start, plot_start + plot_len)],
             real_value[plot_start:plot_start + plot_len],
             color='r', label="ActualValue", linewidth=1, linestyle=linestyle[0])
    plt.plot([xi for xi in range(plot_start, plot_start + plot_len)], real_value[plot_start:plot_start + plot_len], '.',
             markersize=markersize,
             color='r')
    plt.legend(fontsize=fontsize)
    plt.show()
    # plt.savefig(fig_path.format(value_name.format(item_name[attr])))
