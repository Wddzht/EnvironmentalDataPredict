from pylab import *
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime
import numpy as np
import os

color = ['dimgray', 'hotpink', 'yellow', 'r', 'g', 'b', 'orange']
linestyle = ['-', '-', '-', '-', '-', '-', '-']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
window_len = 6
attr_count = 6
plot_len = 600
value_name = '{:} at FuZhou,China'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit'


def calculate_err(real_value, fit_value):
    r2 = r2_score(real_value, fit_value)
    mae = mean_absolute_error(real_value, fit_value)
    rmse = math.sqrt(mean_squared_error(real_value, fit_value))
    return r2, mae, rmse


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

t = datetime.datetime.strptime('2017/1/9 18:00', '%Y/%m/%d %H:%M')  # 福州数据起始时间
# t = datetime.datetime.strptime('2018/12/19 0:00', '%Y/%m/%d %H:%M')  # 波特兰数据起始时间
x_datetime = []
for i in range(len(nation_data)):
    x_datetime.append(t.strftime('%y/%m/%d'))
    t = t + datetime.timedelta(hours=1)

plot_start = x_datetime.index('17/02/01')  # 福州数据绘制起始时间
# plot_start = x_datetime.index('19/02/01')  # 波特兰数据绘制起始时间

x_index = [xi for xi in range(len(x_datetime) + 1) if xi % 48 == 0]
x_datetime = [x_datetime[xi] for xi in range(len(x_datetime) + 1) if xi % 48 == 0]

filenames = os.listdir(file_path)
for attr in range(attr_count):
    real_value = np.array(nation_data)[:, attr]

    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.22, top=0.9)
    # plt.xlabel("Time", fontsize=9.5)
    plt.xticks(x_index, x_datetime, fontsize=9.5, rotation=30)
    plt.plot([xi for xi in range(plot_start, plot_start + plot_len)],
             real_value[plot_start:plot_start + plot_len],
             color=color[0], label="ActualValue", linewidth=1, linestyle=linestyle[0])
    plt.plot([xi for xi in range(plot_start, plot_start + plot_len)], real_value[plot_start:plot_start + plot_len], '.',
             color=color[0])
    plt.title(value_name.format(item_name[attr]))

    # 单位修改，原数据单位统一为 μg/m^3
    if item_name[attr] in ['CO']:
        plt.ylabel("ppm", fontsize=9.5)
    if item_name[attr] in ['PM25', 'PM10']:
        plt.ylabel("μg/m^3", fontsize=9.5)
    color_index = 1
    for filename in filenames:
        fit_file = open(os.path.join(file_path, filename))
        fit_data = []
        if filename[0] == '_':
            continue
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
        plt.plot([xi for xi in range(plot_start, plot_start + plot_len)],
                 fit_data[plot_start:plot_start + plot_len],
                 color=color[color_index],
                 label=filename.split('.')[0],
                 linewidth=1,
                 linestyle=linestyle[color_index])
        plt.plot([xi for xi in range(plot_start, plot_start + plot_len)], fit_data[plot_start:plot_start + plot_len],
                 '.',
                 color=color[color_index])

        plt.legend(fontsize=9)
        color_index += 1
        # r2, mae, rmse = calculate_err(real_value[window_len:len(fit_data)], fit_data[window_len:])
        # print('{:}-{:}\n{:.3}\t{:.3}\t{:.3}\t\n'.format(filename.split('.')[0], item_name[attr], r2, mae, rmse))
        # print('{:.3}\t{:.3}\t{:.3}\t'.format(r2, mae, rmse))
    plt.show()
    # plt.savefig(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_plot\{:}.png'.format(
    #     value_name.format(item_name[attr])))
