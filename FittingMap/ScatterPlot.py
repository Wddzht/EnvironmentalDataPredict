from pylab import *
import numpy as np
import os

color = ['hotpink', 'yellow' ,'dimgray', 'g', 'b', 'orange']
linestyle = ['-', '-', '-', '-', '-', '-', '-']
item_name = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10']
molecular_weight = {'CO': 28, 'NO2': 46, 'SO2': 64, 'O3': 48}  # 分子量
window_len = 6
attr_count = 6
plot_start = 6
plot_len = 2000
fontsize=11
#value_name = '{:} at FuZhou,China'
value_name = '{:} at Portland,USA'
nation_file = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_nation.pare.txt")
file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_prediction_fit'
plt_file_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_scatter_plot\{:}.png'

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
    plt.title(value_name.format(item_name[attr]),fontsize=12)

    # plt.subplots_adjust(bottom=0.22, top=0.9)
    # 单位修改，国内CO:mg/m3，SO2/N O2/O3/颗粒物是ug/m3
    # 波特兰国测站气体单位CO:ppm，SO2/N O2/O3:ppb，颗粒物是ug/m3。
    # mg / m3 = M / 22.4 * ppm，M为气体的分子量
    if item_name[attr] in ['CO']:
        plt.xlabel("Acture value(ppm)", fontsize=fontsize)
        plt.ylabel("Prediction(ppm)", fontsize=fontsize)
    if item_name[attr] in ['SO2', 'NO2', 'O3']:
        plt.xlabel("Acture value(ppb)", fontsize=fontsize)
        plt.ylabel("Prediction(ppb)", fontsize=fontsize)
    if item_name[attr] in ['PM25', 'PM10']:
        plt.xlabel("Acture value(μg/m^3)", fontsize=fontsize)
        plt.ylabel("Prediction(μg/m^3)", fontsize=fontsize)

    real_value = np.array(nation_data)[:, attr]
    max_lim = 0
    # 测量值单位换算（只有福州数据需要）
    if item_name[attr] in ['CO', 'SO2', 'NO2', 'O3']:
        real_value = (molecular_weight[item_name[attr]] / 22.4) * real_value
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

        max_lim = max(max(real_value[plot_start:plot_len]), max(fit_data[plot_start:plot_len]))

        plt.scatter(real_value[plot_start:plot_len], fit_data[plot_start:plot_len],
                    s=8, label=filename.split('.')[0],
                    color=color[color_index])
        # calc the trendline
        z = np.polyfit(real_value[plot_start:plot_len], fit_data[plot_start:plot_len], 1)
        p = np.poly1d(z)
        plt.plot(real_value[plot_start:plot_len], p(real_value[plot_start:plot_len]),
                 color=color[color_index],
                 linewidth=1, linestyle=linestyle[color_index])
        plt.legend(fontsize=fontsize)
        color_index += 1

    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)

    # 对称直线 y = x
    x_line = np.linspace(0, max_lim, plot_len - plot_start)
    plt.plot(x_line, x_line, 'k--')

    # plt.show()
    plt.savefig(plt_file_path.format(value_name.format(item_name[attr])))
