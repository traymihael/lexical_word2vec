# coding: utf-8
import pickle
from collections import OrderedDict
import numpy as np
from scipy import io
import word2vec

fname_input = '../make_corpus/wiki_corpus/wiki_lemma_corpus.txt'
fname_word2vec_out = 'wiki_vectors_lemma.txt'


def make_word2vec():
    # word2vecでベクトル化
    word2vec.word2vec(train=fname_input, output=fname_word2vec_out,
                      size=300, threads=4, binary=0, min_count=1)
    print('finish word2vec')

    # その結果を読み込んで行列と辞書作成
    with open(fname_word2vec_out, 'rt', encoding='utf-8', errors='ignore') as data_file:

        # 先頭行から用語数と次元を取得
        work = data_file.readline().split(' ')
        size_dict = int(work[0])
        size_x = int(work[1])

        # 辞書と行列作成
        dict_index_t = OrderedDict()
        matrix_x = np.zeros([size_dict, size_x], dtype=np.float64)

        with open('out2_normal_lemma.txt', 'w') as out:
            count = 0
            for i, line in enumerate(data_file):
                co  unt += 1
                if count % 10000 == 0:
                    print(count)

                work = line.strip().split()
                dict_index_t[work[0]] = i

                if len(work) == 302:
                    word = work[0] + '_' + work[1]
                    vec = work[2:]
                elif len(work) == 301:
                    word = work[0]
                    vec = work[1:]
                elif len(work) == 303:
                    word = work[0] + '_' + work[1] + '_' + work[2]
                    vec = work[3:]
                else:
                    print(work)
                    a = input()
                matrix_x[i] = vec
                out.write(word + ' ')
                for j in range(len(matrix_x[i])):
                    if j == len(matrix_x[i]) - 1:
                        out.write(vec[j])
                        break
                    out.write(vec[j] + ' ')
                out.write('\n')

        print('finish')


if __name__ == '__main__':
    # word2vec data make
    make_word2vec()
