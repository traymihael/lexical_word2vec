import logging
from gensim.models import word2vec

fname_input = 'text4.txt'
fname_word2vec_out = 'test.model'

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(fname_input)

model = word2vec.Word2Vec(sentences, size=3, min_count=1, window=2, iter=1)
# model = word2vec.Word2Vec(sentences, size=300, min_count=10, window=5)
model.save(fname_word2vec_out)