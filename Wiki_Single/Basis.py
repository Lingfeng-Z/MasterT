import csv
import nltk
import re
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from collections import Counter
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

text= News['SECTION_TEXT']

del News
gc.collect()

X=text.values.tolist()


def majid(X):
    corpus = []
    for i in range(0, len(X)):
        # Normally removing punctuation
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', str(X[i]))  # remove punctuation
        review = re.sub(r'\d+', '', review)  # remove number
        review = review.lower() #lower case
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

del News
del text
gc.collect()
num_cores = 8
sentences = Parallel(n_jobs=num_cores)(delayed(majid)(i) for i in X)


#########

model1 = Word2Vec(sentences, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')

model1.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-CBOW-Basis.bin.gz', binary=True)
model1.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-CBOW-Basis.txt', binary=False)
model1.save('~/MasterThesis/Model/Wiki-W-CBOW-Basis.bin')
print('Done Saving Model1')


#####
model2 = Word2Vec(sentences, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-Skip-Basis.gz', binary=True)
model2.wv.save_word2vec_format('~/MasterThesis/Model/Wiki-W-Skip-Basis.txt', binary=False)
model2.save('~/MasterThesis/Model/Wiki-W-Skip-Basis.bin')
print('Done Saving Model2')


""""
Saving the embeddings and the model

"""""

from gensim.models import Word2Vec, KeyedVectors




# model.save('model2.bin')

print('Done Saving the Embeddings')