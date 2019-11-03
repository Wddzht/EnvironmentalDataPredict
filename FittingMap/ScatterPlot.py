from pylab import *
import numpy as np
import os

color = ['dimgray', 'hotpink', 'yellow', 'r', 'g', 'b', 'orange']
linestyle = ['-', '-', '-', '-', '-', '-', '-']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
window_len = 6
attr_count = 6
plot_start = 600
plot_len = 3000
value_name = '{:} at FuZhou,China'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit'

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
    plt.figure()
    # plt.subplots_adjust(bottom=0.22, top=0.9)
    plt.xlabel("acture value", fontsize=9.5)
    real_value = np.array(nation_data)[:, attr]
    color_index = 1
    for filename in filenames:
        plt.title(value_name.format(item_name[attr]))
        plt.ylabel("prediction of " + filename.split('.')[0], fontsize=9.5)

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
        plt.scatter(real_value[plot_start:plot_len], fit_data[plot_start:plot_len],
                    s=8, label=filename.split('.')[0],
                    color=color[color_index])

        # calc the trendline
        z = np.polyfit(real_value[plot_start:plot_len], fit_data[plot_start:plot_len], 1)
        p = np.poly1d(z)
        plt.plot(real_value[plot_start:plot_len], p(real_value[plot_start:plot_len]),
                 color=color[color_index],
                 linewidth=1, linestyle=linestyle[color_index])
        plt.legend(fontsize=9.5)
        color_index += 1

    plt.show()
    # plt.savefig(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_plot\{:}.png'.format(
    #     value_name.format(item_name[attr])))
