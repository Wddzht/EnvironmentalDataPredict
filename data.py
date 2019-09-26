import pickle
import numpy as np


def read_str_data(data_path, n_seqs=2, window=12, attr_count=12):
    input_file = open(data_path, encoding='utf8')
    data = input_file.readlines()
    seqs_count = len(data) - window - 1
    batch_count = seqs_count // n_seqs
    print("data length: {}\tbatch num: {}".format(len(data), batch_count))

    data_normalized = []
    # 归一化
    max = np.zeros(attr_count)
    min = np.zeros(attr_count)

    d0_list = data[0].split()
    for i in range(len(d0_list)):
        max[i] = min[i] = float(d0_list[i])

    for item in data:
        _item = item.split()
        line = []
        i = 0
        for d in _item:
            line.append(float(d))
            if float(d) > max[i]:
                max[i] = float(d)
            elif float(d) < min[i]:
                min[i] = float(d)
            i += 1
        data_normalized.append(line)

    for i in range(len(data_normalized)):
        for d in range(len(data_normalized[0])):
            data_normalized[i][d] = (data_normalized[i][d] - min[d]) / (max[d] - min[d])

    index = 0
    seqs = []
    targets = []
    while len(seqs) < seqs_count:
        seq = []
        for j in range(window):
            if len(data_normalized[index + j]) != attr_count:
                raise ('Err:' + str(data_normalized[index + j]))
            for item in data_normalized[index + j]:
                seq.append(item)
        seqs.append([seq])

        seq_target = []
        for item in data_normalized[index + window]:
            seq_target.append(item)
        targets.append([seq_target])

        index += 1

    index_b = 0
    batches = []
    for k in range((len(seqs) // n_seqs)):
        batche_seqs = []
        batche_targets = []
        while index_b < (len(seqs) // n_seqs) * n_seqs:
            if len(batche_seqs) < n_seqs:
                batche_seqs.append(seqs[index_b])
                batche_targets.append(targets[index_b])
                index_b += 1
            else:
                batches.append([batche_seqs, batche_targets])
                batche_seqs = []
                batche_targets = []

    return batches, min, max
