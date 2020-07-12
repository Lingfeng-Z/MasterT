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
import sys



csv.field_size_limit(sys.maxsize)
News = pd.read_csv("~/MasterThesis/Data/wiki/wiki.csv", sep=',',engine = 'python',iterator=True)
loop = True
chunkSize = 1000
chunks = []
index=0
while loop:
    try:
        print(index)
        chunk = News.get_chunk(chunkSize)
        chunks.append(chunk)
        index+=1
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print('开始合并')
News = pd.concat(chunks, ignore_index= True)

print('Done Importing')

text = News['SECTION_TEXT']
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
filepath = '~/MasterThesis/Data/dict/dic-wiki.csv'
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


def majid2(X):
    sentences = replace_negations(X)
    return sentences


num_cores = multiprocessing.cpu_count()
sent2 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sentences)

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

model1.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-CBOW-Neg.bin.gz', binary=True)
model1.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-CBOW-Neg.txt', binary=False)
model1.save('~/MasterThesis/Model/Wiki-W-CBOW-Neg.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-Skip-Neg.gz', binary=True)
model2.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-Skip-Neg.txt', binary=False)
model1.save('~/MasterThesis/Model/Wiki-W-Skip-Neg.bin')
print('Done Saving Model2')

# model.save('model2.bin')

print('Done Saving the Embeddings')







