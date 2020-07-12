from gensim.models import word2vec


#Basic
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Basis.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Basis.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))

#STEM
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Stem.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Stem.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))

#Spell
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Spell.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Spell.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))

#Punc
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Punc.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Punc.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))

#Stop
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-Stop.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-Stop.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))


#POS
model1 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-CBOW-POS.bin')
model2 = word2vec.Word2Vec.load('/home/lingfengzhang/Code/Sync/MasterThesis/Model/W-Skip-POS.bin')
SizeOfVocab1 = model1.wv.vocab
SizeOfVocab2 = model2.wv.vocab
print('Size of Vocabulary=', len(SizeOfVocab1))
print('Size of Vocabulary=', len(SizeOfVocab2))
