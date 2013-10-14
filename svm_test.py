#encoding=utf-8

import jieba
import jieba.analyse
import re
import codecs
from gensim import corpora,models,similarities
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dictionary = corpora.Dictionary.load('/tmp/douban_book_music.dict')
lsi = models.LsiModel.load('/tmp/douban_book_music.lsi')

f = open("book_1315.txt", "r")
doc=[]

for line in f:
  re.match(u'\u4f60',u'\u4f60')
  cline= line.decode('utf-8')
  p =re.compile(u'[^\u4e00-\u9fa5]')
  c = p.sub('',cline)    # type(c) is str
  doc =doc + [c]


stoplist = set(u'的 吗 了 也 吧 还 你 我 他 您'.split())      #remove common word
ptext= [' '.join(jieba.cut(line)) for line in doc]        #words are spiltted by empty space, like english word                
texts1=[[word for word in line.split() if word not in stoplist] for line in ptext]
texts=[[word for word in line if len(word)>=2] for line in texts1]    #delete the word which length is less than 2

# for line in texts:
#     for word in line:
#         if len(word)<2:
#             line.remove(word)
#             #print word
#         else:
#             pass

#dictionary= corpora.Dictionary(texts)
#dictionary.save('/tmp/douban_book_music.dict')
#print dictionary
corpus =[dictionary.doc2bow(text) for text in texts]  #corpus is a sparse-matrix (doc~word matrix), type is list
#corpora.MmCorpus.serialize('/tmp/douban_book_music.mm', corpus)
tfidf= models.TfidfModel(corpus)                      #tfidf is a python object, input is vector/matrix
corpus_tfidf=tfidf[corpus]                            #sparse-matrix weighted
#corpora.MmCorpus.serialize('/tmp/douban_book_music_test.mm', corpus_tfidf)
# #lda.print_topics(3) is list type
# lda=models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=100)
# lda.save('/tmp/model.lda')
# tmp=lda.print_topics(100)
# #tmp=lda.print_topic(1,50)
# # lda1=[line.decode('utf-8') for line in tmp]


# lsi=models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=100)
# lsi.save('/tmp/douban_book_music.lsi')
#tmp=lsi.print_topics(100)
corpus_lsi = lsi[corpus_tfidf]
corpora.MmCorpus.serialize('/tmp/douban_book_music_lsi_test.mm', corpus_lsi)



