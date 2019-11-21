attrs = ['CO', 'NO2', 'SO2', 'O3', 'PM25', 'PM10', 'TEMPERATURE', 'HUMIDITY', 'PM05N', 'PM1N', 'PM25N', 'PM10N']

file = open(r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_micro.pare.txt')
save_path = r'E:\_Python\ScipPredictor\EnvironmentalDataPredict\FittingMap\data\ptl_micro_Del_SO2.pare.txt'
output = open(save_path, 'w')

for line in file:
    items = line.strip().split()
    new = []
    for i in range(len(attrs)):
        if attrs[i] in ['CO', 'NO2', 'O3', 'PM25', 'PM10', 'TEMPERATURE', 'HUMIDITY', 'PM05N', 'PM1N', 'PM25N', 'PM10N']:
            new.append(items[i])
    output.write('\t'.join(new) + '\n')
