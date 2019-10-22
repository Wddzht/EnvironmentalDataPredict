from pylab import *
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime
import numpy as np
import os

color = ['k', 'hotpink', 'yellow', 'r', 'g', 'b', 'orangered']
linestyle = ['-', '-', '-', '-', '-', '-', '-']
mpl.rcParams['font.sans-serif'] = ['SimHei']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
window_len = 6
attr_count = 6
plot_len = 800
value_name = '福州{:}预测曲线'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_prediction_fit'


def calculate_err(real_value, fit_value):
    r2 = r2_score(real_value, fit_value)
    mae = mean_absolute_error(real_value, fit_value)
    rmse = math.sqrt(mean_squared_error(real_value, fit_value))
    return r2, mae, rmse


t = datetime.datetime.strptime('2017/1/9 18:00', '%Y/%m/%d %H:%M')  # 福州数据起始时间
# t = datetime.datetime.strptime('2018/12/19 0:00', '%Y/%m/%d %H:%M')  # 波特兰数据起始时间
x = []
for i in range(800):
    x.append(t.strftime('%y/%m/%d'))
    t = t + datetime.timedelta(hours=1)

nation_data = []
for line in nation_file:
    line = line.strip()
    if not line:
        continue
    _list = []
    for i in line.split('\t'):
        _list.append(float(i))
    nation_data.append(_list)

filenames = os.listdir(file_path)
for attr in range(attr_count):
    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(bottom=0.22, top=0.9)
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("μg m^-3", fontsize=10)
    plt.xticks([xi for xi in range(len(x) + 1) if xi % 48 == 0],
               [x[xi] for xi in range(len(x) + 1) if xi % 48 == 0],
               fontsize=10, rotation=30)
    real_value = np.array(nation_data)[:, attr]
    plt.plot([xi for xi in range(window_len, plot_len)],
             real_value[window_len:plot_len],
             color=color[0], label="ActualValue", linewidth=2, linestyle=linestyle[0])
    plt.title(value_name.format(item_name[attr]))
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
        plt.plot([xi for xi in range(window_len, plot_len)],
                 fit_data[window_len:plot_len],
                 color=color[color_index],
                 label=filename.split('.')[0],
                 linewidth=2,
                 linestyle=linestyle[color_index])

        plt.legend(fontsize=10)
        color_index += 1
        r2, mae, rmse = calculate_err(real_value[window_len:len(fit_data)], fit_data[window_len:])
        # print('{:}-{:}\n{:.3}\t{:.3}\t{:.3}\t\n'.format(filename.split('.')[0], item_name[attr], r2, mae, rmse))
        print('{:.3}\t{:.3}\t{:.3}\t'.format(r2, mae, rmse))
    #plt.show()
    plt.savefig(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_plot\{:}.png'.format(
        value_name.format(item_name[attr])))
