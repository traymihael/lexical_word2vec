import numpy as np

def get_pair(train, val, _, vocab):
    text_name = 'mi_data.txt'

    original_index = []
    # concat_index = []
    ori_con_data = []
    data_mi = []
    train = list(train)
    val = list(val)
    _ = list(_)
    start = len(vocab)

    with open(text_name, 'r', encoding = 'utf-8',errors = 'ignore') as f:
        for line in f:
            data = line.strip().split('\t')
            data_mi.append([start])
            for i in range(len(data)-1):
                train.append(start)
                val.append(start)
                _.append(start)
                original_index.append(int(data[i+1]))
                data_mi[-1].append(data[i+1])
                # concat_index.append(str(start))
            ori_con_data.append([start, int(train[int(data[1])])])
            vocab.update({data[0]: start})
            start += 1

    # with open('concat_ind.txt', 'w') as out:
    #     out.write('\t'.join(concat_index))
    with open('data_mi.txt', 'w') as out:
        for i in range(len(data_mi)):
            if len(data_mi[i]) >= 10000:
                for j in range(int(len(data_mi[i])/10000)+1):
                    if j == 0:
                        data_mi_kari = list(map(str, data_mi[i][j*10000:(j+1)*10000]))
                    else:
                        data_mi_kari = list(map(str, data_mi[i][j*10000 - 1:(j+1)*10000]))
                        data_mi_kari[0] = str(data_mi[i][0])
                    out.write('\t'.join(data_mi_kari))
                    out.write('\n')
            else:
                data_mi_kari = list(map(str, data_mi[i]))
                out.write('\t'.join(data_mi_kari))
                out.write('\n')

    train.append(start)
    val.append(start)
    _.append(start)
    vocab.update({'xxxxxxxx': start})

    train = np.array(train)
    val = np.array(val)
    _ = np.array(_)

    # concat_indは後ろにつくインデックス
    # originalindは派生前のインデックス
    return  train, val, _, vocab, original_index, ori_con_data
