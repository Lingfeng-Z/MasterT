import nltk.data
import pandas as pd
import nltk.data
import csv
import nltk
from nltk.corpus import wordnet
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
text= News['SECTION_TEXT']
X=text.values.tolist()


def majid(X):
    corpus = []
    for i in range(0, len(X)):
        #review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', str(X[i]))  # remove punctuation
        review = re.sub(r'\d+', ' ', str(X[i]))  # remove number
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

from multiprocessing import Process, freeze_support


word_map = {}
ukWac = '~/MasterThesis/Data/ukwac/sorted.uk.word.unigrams.csv'
word_freq = {}
with open(ukWac, encoding="utf8") as f:
    for line in f:
        line = line.rstrip()
        if line:
            x = line.split(',')
            #print(key, val)
            word_freq[x[0]] = int(x[-1])


def create_dic(sent, word_map):
    antonyms = set()
    i, l = 0, len(sent)
    indices = [i for i, x in enumerate(sent) if x == 'not' or x == 'never' or x == 'nor' or x == 'neither']
    for i in indices:
        if i+1 < l:
            word = sent[i+1]
            if word not in word_map:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        for antonym in lemma.antonyms():
                            antonyms.add(antonym.name())
                if len(antonyms) == 1:
                    #print(antonyms)
                    gc.collect()
                    word_map[word] = antonyms.pop()
                elif len(antonyms) > 1:
                    temp_freq = 0
                    target_antonym = 'None'
                    for antonym in antonyms:
                        if antonym in word_freq.keys():
                            if word_freq[antonym] > temp_freq:
                                target_antonym = antonym
                                temp_freq = word_freq[antonym]
                    if target_antonym == 'None':
                        target_antonym = antonyms.pop()
                    word_map[word] = target_antonym
                else:
                    word_map[word] = 'None'


gc.collect()

import multiprocessing
from multiprocessing import Process, freeze_support,  Manager

def create_dic_multiprocessor(d,sent):
    i = 0
    #word_map = {}
    #print(sent)
    for x in sent:
        create_dic(x, d)
        i += 1
        if (i % 1000) == 0:
            print(i)

processes = []
manager = Manager()
d = manager.dict()
length = len(sentences)
for i,v in [( int(j*(length/num_cores)), int((j+1)*length/num_cores)) for j in range(num_cores)]:
    sent = sentences[i:v]
    p = multiprocessing.Process(target=create_dic_multiprocessor, args=(d, sent))
    processes.append(p)
    p.start()

for process in processes:
    process.join()

for k,v in d.items():
    word_map[k] = v


#write to dic
w = csv.writer(open("~/MasterThesis/Data/dict/dic-wiki.csv", "w",newline='', encoding = 'utf-8'))
for key, val in word_map.items():
    #print(key, val)
    w.writerow([key, val])