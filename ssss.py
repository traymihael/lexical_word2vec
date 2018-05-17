
import word2vec
fname_input = 'L1.txt'
fname_word2vec_out = 'test.model'

# word2vecでベクトル化
import sys
# for line in sys.path:
#     print(line)
word2vec.word2vec(train=fname_input, output=fname_word2vec_out,size=300, threads=1, binary=0)
print('finish word2vec')
