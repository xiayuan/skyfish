#encoding=utf-8
import jieba
import jieba.analyse
import re
import codecs
from gensim import corpora,models,similarities
import gensim
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


##SVM training part
corpus_lsi = corpora.MmCorpus('tmp/douban_book_music_lsi.mm')
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus_lsi)   #100*4000 dimension

clf = svm.SVC(probability=True)
X=scipy_csc_matrix.transpose()                        #4000*100
print X.shape
a=np.zeros(2000)
b=np.ones(2000)
y=np.hstack((a, b))
y.shape=(4000,1)                                      #4000*1
print y.shape
# clf.fit(X, y)
# joblib.dump(clf, 'tmp/svm.pkl')



##SVM testing part
dictionary = corpora.Dictionary.load('tmp/douban_book_music.dict')
lsi = models.LsiModel.load('tmp/douban_book_music.lsi')

ft=open('test.txt','w')

f = open("book_1.txt", "r").readline()
ft.write(f)
ft.write("\n"+ "我是没用的词，啦啦啦，算法导论")
ft.close()

f = open("test.txt", "r")

doc=[]

for line in f:
  re.match(u'\u4f60',u'\u4f60')
  cline= line.decode('utf-8')
  p =re.compile(u'[^\u4e00-\u9fa5]')
  c = p.sub('',cline)    # type(c) is str
  doc =doc + [c]
f.close()


stoplist = set(u'的 吗 了 也 吧 还 你 我 他 您'.split())      #remove common word
ptext= [' '.join(jieba.cut(line)) for line in doc]        #words are spiltted by empty space, like english word                
texts=[[word for word in line.split() if word not in stoplist] for line in ptext]

corpus =[dictionary.doc2bow(text) for text in texts]  #corpus is a sparse-matrix (doc~word matrix), type is list
tfidf= models.TfidfModel(corpus)                      #tfidf is a python object, input is vector/matrix
corpus_tfidf=tfidf[corpus]                            #sparse-matrix weighted

corpus_lsi = lsi[corpus_tfidf]


# corpora.MmCorpus.serialize('/tmp/douban_book_music_lsi_test_1.mm', corpus_lsi)
# corpus_lsi_test = corpora.MmCorpus('/tmp/douban_book_music_lsi_test_1.mm')
scipy_csc_matrix_test = gensim.matutils.corpus2dense(corpus_lsi,100)   #100*1315 dimension
T=scipy_csc_matrix_test

print T.shape
T=scipy_csc_matrix_test.transpose()                        #1315*100

length= T.shape[0]

# l=np.zeros(T.shape[0])
# l.shape=(T.shape[0],1)
#print(clf.predict(T)[500:1300])
clf_t= joblib.load('tmp/svm.pkl')

print clf_t.predict_proba(T)[0:length-1]
print clf_t.predict(T)[0:length-1]



fr = open("result.txt", "w")
fr.write('probability book/music: '+ str(clf_t.predict_proba(T)[0:length-1]) + '\n')
if int(clf_t.predict(T)[0:length-1])==0:
    fr.write('Classified as BOOK')
if int(clf_t.predict(T)[0:length-1])==1:
    fr.write('Classified as MUSIC')
fr.close()
#print clf_t.score(T, l)
#print clf_t

