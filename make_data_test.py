import logging
from gensim.models import word2vec

fname_input = 'L1.txt'
fname_word2vec_out = 'test_test.model'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(fname_input)

model = word2vec.Word2Vec(sentences, size=3, min_count=1, window=4)
model.save(fname_word2vec_out)