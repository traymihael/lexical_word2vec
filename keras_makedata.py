# import db
#
# def text2doc(text):
#     results = tagger.parse(text).split('\n')
#     doc = [result.token
#         for result in map(MecabResult, results)
#         if not result.POS[0] in ['記号', '助詞', '助動詞', None]]
#     return doc

# docs = [text2doc(song['lyric'], 'mecab') for song in list(db.songs.find())]
docs = [['I', 'have', 'a', 'pen'], ['I', 'have','an','apple']]

dic = Dictionary(docs)
dic.id2token = {val:key for key, val in dic.token2id.items()}
dic.filter_extremes(no_below=3, no_above=0.3)