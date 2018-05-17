def search(text_name):
    dict = [['apple_pine'], ['orange_pine']]
    with open(text_name, 'r', encoding = 'utf-8',errors = 'ignore') as f:
        count = 0
        for line in f:
            line = line.strip().split()
            if count < 0:
                count += len(line)
            for i in range(len(line)):
                if line[i] == 'apple':
                    dict[0].append(str(count+i))
                    dict[0] += [str(9) for j in range(20000)]
                if line[i] == 'orange':
                    dict[1].append(str(count+i))
            count += len(line) + 1

    return dict

def write_data(dict, out_name):
    with open(out_name, 'w') as out:
        for line in dict:
            out.write('\t'.join(line))
            out.write('\n')

if __name__ == '__main__':
    text_name = 'text6.txt'
    out_name = 'mi_data.txt'
    dict = search(text_name)
    write_data(dict, out_name)