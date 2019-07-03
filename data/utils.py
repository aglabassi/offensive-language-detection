#!/usr/bin/env python3
#Contains helper classes and functions for preprocessing
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:04:34 2019

@author: Abdel Ghani Labassi
"""

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as sw
import pandas as pd 
import re
from bs4 import BeautifulSoup
from sklearn.utils import shuffle

#A custom tokenizer
class TwitterPreprocessorTokenizer():
    
    def __init__(self, stem=True, stopwords=True):
        
        self.stopwords =  set(sw.words('english')) if stopwords else set()
        
        self.stem = SnowballStemmer("english",True).stem if stem else lambda x:x
            
        
    def __call__(self, text):  
        
        cleaned_text =  TwitterPreprocessorTokenizer._clean(str(text))
        
        return [ self.stem(word) for word in word_tokenize(cleaned_text) if word not in self.stopwords ]
    
    
    def _clean(document):
        
        #HTML decoding
        cleaned_document = BeautifulSoup(document,'lxml').get_text()
        
        #Remove urls
        cleaned_document = re.sub('https?://[A-Za-z0-9./]+','', document)
        
        #Remove @Mention
        cleaned_document = re.sub(r'@[A-Za-z0-9]+','', cleaned_document)
        
        #UTF-8 BOM decoding
        try:
            cleaned_document = cleaned_document.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            pass
        
        #Keep just the letter and ! and ?
        #cleaned_document = re.sub("[^!|?|a-zA-Z]", " ", cleaned_document)

        #To lower case        
        cleaned_document = cleaned_document.lower()
        
        return cleaned_document
    
  
    #Builds a vocab 
    def build_vocab(self, texts): 
        
        tokenized_texts = [ self(text) for text in texts ]
        
        vocab = [word for tokenized_text in tokenized_texts for word in tokenized_text]
        
        #Index 0 contains empty word
        vocab = [""] + vocab
        
        return vocab
    
 
                
#returns (corpus_train,corpus_test,y_train, y_test), with corpus in python's list format, 
#ignoring unlabeled documents. Assume dataset is a dataframe from the imbd dataset csv file    
def  get_dataset_from_IMDB( file_path ):
    
    df = pd.read_csv(file_path, encoding = "ISO-8859-1")
        
    df = df[ ['review', 'label'] ]

    df = df[ df['label'] != 'unsup' ]
    
    df['label'] = df['label'].map({'pos': 1, 'neg': 0})
    
    corpus = df['review']
    y = df['label']
    
    return corpus, y

def get_twitter_dataset():
    
    #Getting data from first file, 30% offensive
    df1 = pd.read_csv( "../data/tweet.tsv", sep = "\t" )
    pos = df1[ df1["target"] == 1 ]
    neg = df1[ df1["target"] == 0 ]
    
    df1 = pd.concat([ pos, neg.iloc[:len(pos),] ])
    
    #Getting data from second file, 50% offensive
    df2 = pd.read_csv("../data/tweet2.csv",encoding = "ISO-8859-1")
    df2.rename(columns={'does_this_tweet_contain_hate_speech':'target',
                          'tweet_text':'input'}, inplace=True)
    
    df2["target"] = df2["target"].replace("The tweet uses offensive language but not hate speech", 1)
    df2["target"] = df2["target"].replace("The tweet contains hate speech", 1)
    df2["target"] = df2["target"].replace("The tweet is not offensive", 0)
    
    df2 = df2[ ["input", "target"] ]
    
    #Combining both files
    df= pd.concat( [df1, df2] )
    
    return shuffle(df)

def make_twitter_dataset():
    df = get_twitter_dataset()
    df.to_csv("../data/tweet_old.csv")


