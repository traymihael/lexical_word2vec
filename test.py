# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text6.txt')

model = word2vec.Word2Vec(sentences, size=300, min_count=1, window=5)

model.save("jawiki_wakati.model")