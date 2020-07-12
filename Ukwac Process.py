import pandas as pd
import csv
import re

ukWac = '/home/lingfengzhang/Code/Sync/MasterThesis/Data/ukwac/sorted.uk.word.unigrams'
word_freq = {}
w = csv.writer(open("/home/lingfengzhang/Code/Sync/MasterThesis/Data/ukwac/sorted.uk.word.unigrams.csv", "w",newline='', encoding = 'utf-8'))
with open(ukWac, mode = 'rb') as f:
    for line in f:
        line = line.decode(errors='ignore')
        line = re.sub('\n', '', line)
        line = line.split('\t')
        print([line[1], line[0]])
        w.writerow([line[1], int(line[0])])
