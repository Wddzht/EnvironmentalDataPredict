# 按时间匹配两个数据，前提数据是按时间升序的。输出数据不包含时间
import time

file1 = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.txt")
file2 = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.txt")

output1 = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_micro.pare.txt", "w")
output2 = open(r"E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\fz_nation.pare.txt", "w")

line2 = file2.readline()
time2 = line2.split('\t')[0]
for line1 in file1:
    time1 = line1.split('\t')[0]
    if time.strptime(time1, '%Y/%m/%d %H:%M') < time.strptime(time2, '%Y/%m/%d %H:%M'):
        continue
    elif time1 == time2:
        output1.write('\t'.join(line1.split('\t')[1:]))
        output2.write('\t'.join(line2.split('\t')[1:]))
    else:
        for line2 in file2:
            time2 = line2.split('\t')[0]
            if time.strptime(time1, '%Y/%m/%d %H:%M') < time.strptime(time2, '%Y/%m/%d %H:%M'):
                break
            elif time1 == time2:
                output1.write('\t'.join(line1.split('\t')[1:]))
                output2.write('\t'.join(line2.split('\t')[1:]))
                break
            elif time.strptime(time1, '%Y/%m/%d %H:%M') > time.strptime(time2, '%Y/%m/%d %H:%M'):
                continue

file1.close()
file2.close()
output1.close()
output2.close()
