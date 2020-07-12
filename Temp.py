import pandas as pd
import csv
import re
from nltk.corpus import wordnet
import joblib
import sys


csv.field_size_limit(sys.maxsize)
data = pd.read_csv("/home/lingfengzhang/Code/Sync/MasterThesis/Data/wiki/wiki.csv", sep=',',engine = 'python',iterator=True)
loop = True
chunkSize = 1000
chunks = []
index=0
while loop:
    try:
        print(index)
        chunk = data.get_chunk(chunkSize)
        chunks.append(chunk)
        index+=1
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
print('开始合并')
data = pd.concat(chunks, ignore_index= True)
print(data)
