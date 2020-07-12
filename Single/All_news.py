import csv
import nltk
import re
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
import gc
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import multiprocessing
import pickle
import joblib

News = pd.read_csv("/home/lingfengzhang/Code/Sync/MasterThesis/Data/news/news.csv")
print('Done Importing')

News['content'] = News['content'].apply(lambda x: x.lower())

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many',
                'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does',
                'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',
                'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


News['content'] = News['content'].apply(lambda x: correct_spelling(x, mispell_dict))

contraction_mapping = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he has",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


News['content'] = News['content'].apply(lambda x: clean_contractions(x, contraction_mapping))

print("Done Spell")
##########
#Punc
##########
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text

News['content'] = News['content'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

print("Done Punc")
##########
#Basis
##########

text= News['content']
X=text.values.tolist()

def majid(X):
    corpus = []
    for i in range(0, len(X)):
        # Normally removing punctuation
        #review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', str(X[i]))  # remove punctuation
        review = re.sub(r'\d+', '', str(X[i]))  # remove number
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

num_cores = multiprocessing.cpu_count()
sentences = Parallel(n_jobs=num_cores)(delayed(majid)(i) for i in X)
print("Done Basis")
##########
#Negation Handling
##########
gc.collect()
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


def majid1(sent):
    res = []
    for X in sent:
        sentences = replace_negations(X)
        res.append(sentences)
    return res


processes = []
q = multiprocessing.Queue()
length = len(sentences)
pool = multiprocessing.Pool(processes=int(num_cores/2))
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
    sent2 = sent2 + pool.apply_async(majid1, args=(sent,)).get()
pool.close()
pool.join()

print("Done Neg")

##########
#POS
##########
def pos_tagger(sentences):
    tags = []  # have the pos tag included
    nava_sen = []
    pt = nltk.pos_tag(sentences)
    #     for s in sentences:
    #     s_token = nltk.word_tokenize(sentences)
    #     pt = nltk.pos_tag(s_token)
    nava = []
    nava_words = []
    for t in pt:
        if t[1].startswith('NN') or t[1].startswith('NNS') or t[1].startswith('NNP') or t[1].startswith('NNPS') or t[
            1].startswith('JJ') or t[1].startswith('JJR') or t[1].startswith('JJS') or t[1].startswith('VB') or t[
            1].startswith('VBG') or t[1].startswith('VBN') or t[1].startswith('VBP') or t[1].startswith('VBZ') or t[
            1].startswith('RB') or t[1].startswith('RBR') or t[1].startswith('RBS'):
            nava.append(t)
            nava_words.append(t[0])
    return nava_words


def majid2(X):
    review = pos_tagger(X)
    gc.collect()
    return review

num_cores = multiprocessing.cpu_count()
sent3 = Parallel(n_jobs=num_cores)(delayed(majid2)(i) for i in sent2)
print("Done POS")
##########
#Stopwords
##########
stopwords_list = stopwords.words('english')

def remove_stopwords(sentences):
    #        words = sentences.split()
    clean_words = [word for word in sentences if (word not in stopwords_list)]
    return clean_words


# sentences= remove_stopwords(sent)

def majid3(X):
    sentences = remove_stopwords(X)
    gc.collect()
    return sentences


# X = [[el] for el in X]

num_cores = multiprocessing.cpu_count()
sent4 = Parallel(n_jobs=num_cores)(delayed(majid3)(i) for i in sent3)
print("Done Stop")
del sent2
del sent3
del stopwords_list
del word_map
gc.collect()
joblib.dump(sent4, '/home/lingfengzhang/Code/Sync/MasterThesis/Temp/sent4', compress=3)
##########
#Stem
##########
gc.collect()
sent4 = joblib.load('/home/lingfengzhang/Code/Sync/MasterThesis/Temp/sent4')

def stemming2(sentences):
    sno = nltk.stem.SnowballStemmer('english')
    # words = sent.split()
    stemmed_words = [sno.stem(word) for word in sentences]
    return stemmed_words


def majid4(X):
    sentences = stemming2(X)
    gc.collect()
    return sentences

# X = [[el] for el in X]

num_cores = multiprocessing.cpu_count()
sent5 = Parallel(n_jobs=num_cores)(delayed(majid4)(i) for i in sent4)
print("Done Stem")
print("Done All")
##########
#Save
##########

model1 = Word2Vec(sent5, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=0)
print('Done Training')

SizeOfVocab = model1.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')

#####
model2 = Word2Vec(sent5, min_count=5, size=300, workers=multiprocessing.cpu_count(), window=1, sg=1)
print('Done Training')

SizeOfVocab = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab))
print('Done making the Vocabulary')


model1.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-ALL.bin.gz', binary=True)
model1.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-ALL.txt', binary=False)
model1.save('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-ALL.bin')
print('Done Saving Model1')
#####
model2.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-ALL.gz', binary=True)
model2.wv.save_word2vec_format('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-ALL.txt', binary=False)
model1.save('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-ALL.bin')
print('Done Saving Model2')

# model.save('model2.bin')

print('Done Saving the Embeddings')