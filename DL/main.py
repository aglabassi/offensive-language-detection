#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:25:04 2019

@author: Abdel
"""

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_utils", "../data/utils.py" )


import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import pandas as pd
import gensim
from sklearn.metrics import accuracy_score
from data_utils import make_twitter_dataset, TwitterPreprocessorTokenizer
from utils import train_model, narray_to_tensor_dataset, raw_texts_to_sequences
from CNN import CNN4old

#Hyperparameters
SHUFFLE_DATASET = False
EPOCHS = 3
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
NB_KRNL = 128
KRNL_WINDOW_SIZES = [2,3]
DROP1 = 0.3
DROP2 = 0.3

#Geting the dataset
if SHUFFLE_DATASET:
    make_twitter_dataset()
    
df = pd.read_csv( "../data/tweet_old.csv" )
X_raw, y = df["input"], df["target"]

#Getting pretrained embeddings and vocab
model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/glove.twitter.27B.100d.txt')
vocab = model.vocab

#Init tokenizer
tokenizer = TwitterPreprocessorTokenizer(stopwords=False, stem=False)

#transforming  raw texts as sequences of indexes
X = raw_texts_to_sequences(X_raw, tokenizer=tokenizer, vocab=vocab)

#Train/test split
split_loc = int(TRAIN_SIZE * len(X))
X_train, X_test = X[:split_loc], X[split_loc:]
y_train, y_test = y.iloc[:split_loc], y.iloc[split_loc:]

#Tensorifying
train, test =  narray_to_tensor_dataset(X_train, X_test, y_train, y_test)
trainloader = DataLoader(train, batch_size=BATCH_SIZE)


#Initialising model
net = CNN4old(len(vocab), emb_weights=torch.FloatTensor(model.vectors), nb_krnl=NB_KRNL,  window_sizes=KRNL_WINDOW_SIZES, drop1=DROP1, drop2=DROP2)
criterion = BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


#Training the model
print(net)
for epoch in range(EPOCHS):    
    epoch_loss, epoch_acc, epoch_loss_valid, epoch_acc_valid = train_model(net, trainloader, criterion, optimizer, accuracy_score, test)
    print("epoch: ",epoch+1, "/", EPOCHS, "  batch_size: ", BATCH_SIZE )
    print(f'Train_Loss: {epoch_loss:.3f} | Train_acc: {epoch_acc*100:.2f}%')
    print(f'Valid_Loss: {epoch_loss_valid:.3f} | Valid_acc: {epoch_acc_valid*100:.2f}%')
    print('---------------------------')

