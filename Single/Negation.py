"""
@author: Nastaran Babanejad
"""

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk.data
import csv
import nltk
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset, Lemma
import re
import gc



News = pd.read_csv("/home/lingfengzhang/Code/Sync/MasterThesis/Data/news/news.csv")

print('Done Importing')

text = News['content']
X = text.values.tolist()


def majid(X):
    corpus = []
    for i in range(0, len(X)):
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', str(X[i]))  # remove punctuation
        review = re.sub(r'\d+', ' ', review)  # remove number
        review = review.lower()  # lower case
        review = re.sub(r'\s+', ' ', review)  # remove extra space
        review = re.sub(r'<[^>]+>', '', review)  # remove Html tags
        review = re.sub(r'\s+', ' ', review)  # remove spaces
        review = re.sub(r"^\s+", '', review)  # remove space from start
        review = re.sub(r'\s+$', '', review)  # remove space from the end
        corpus.append(review)
    #    return corpus
    # Tokenizing and Word Count
    words = []
    for i in range(len(corpus)):
        words = nltk.word_tokenize(corpus[i])
        # sentences.append(words)

    return words


X = [[el] for el in X]

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
sentences = Parallel(n_jobs=num_cores)(delayed(majid)(i) for i in X)

import time
import multiprocessing
from multiprocessing import Process, freeze_support

len(sentences)

# load dictionary
filepath = '/home/lingfengzhang/Code/Sync/MasterThesis/Data/dict/dic-news.csv'
word_map = {}
with open(filepath, encoding="utf8") as f:
    for line in f:
        line = line.rstrip()
        if line:
            x = line.split(',')
            print(x)
            # print(key, val)
            word_map[x[0]] = str(x[1])

import re


def replace_negations(sent):
    idx = 0
    indices = []
    indices = [i for i, x in enumerate(sent) if x in ["not", "nor", "neither", "never"]]
    # print(indices)
    c = 0
    for i in indices:
        l = len(sent)
        i -= c
        # print(word)
        if i >= 0 and i + 1 < l:
            if sent[i + 1] in word_map and word_map[sent[i + 1]] != 'None':
                sent[i] = word_map[sent[i + 1]]
                sent.pop(i + 1)
                c += 1
    return sent


gc.collect()

len(sentences)

import time
import multiprocessing
from multiprocessing import Process, freeze_support, Manager

from joblib import Parallel, delayed
import multiprocessing


def majid2(sent):
    res = []
    for X in sent:
        sentences = replace_negations(X)
        res.append(sentences)
    return res


processes = []
q = multiprocessing.Queue()
length = len(sentences)
pool = multiprocessing.Pool(processes=num_cores)
'''
for i,v in [( int(j*(length/num_cores)), int((j+1)*length/num_cores)) for j in range(num_cores)]:
    sent = sentences[i:v]
    p = multiprocessing.Process(target=majid2, args=(q, sent))
    processes.append(p)
    p.start()
'''
sent2 = []
for i,v in [( int(j*(length/num_cores)), int((j+1)*length/num_cores)) for j in range(num_cores)]:
    sent = sentences[i:v]
    sent2 = sent2 + pool.apply_async(majid2, args=(sent,)).get()
pool.close()
pool.join()


'''
for process in processes:
    process.join()


sent2 = []
for i in processes:
    sent2 = sent2 + q.get()
'''
#########

model1 = Word2Vec(sent2, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sent2, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')

""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors

model1.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Neg.bin.gz', binary=True)
model1.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Neg.txt', binary=False)
model1.save('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Neg.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Neg.gz', binary=True)
model2.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Neg.txt', binary=False)
model1.save('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Neg.bin')
print('Done Saving Model2')

# model.save('model2.bin')

print('Done Saving the Embeddings')







