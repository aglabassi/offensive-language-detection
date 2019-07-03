#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  20 14:33:51 2019

@author: Abdel Ghani Labassi
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

def load_src(name, fpath):
    import os, imp
    return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("utils", "../data/utils.py" )

from utils import get_twitter_dataset, PreprocessorTokenizer
import genetic_algorithm as ga


# Getting the dataset, separate training / test examples
dataset = get_twitter_dataset()
Xraw_train, Xraw_test, y_train, y_test = train_test_split(dataset["input"], dataset["target"], train_size=0.8)

# Feature extraction
vectorizer = TfidfVectorizer(tokenizer=PreprocessorTokenizer(), 
                             strip_accents='unicode', 
                             use_idf=True, 
                             lowercase=True, 
                             stop_words='english')

X_train = vectorizer.fit_transform(Xraw_train)
X_test = vectorizer.transform(Xraw_test)


## Feature selection
##
##Using Chi2
#
#chi2_selector = SelectKBest( chi2, k=1000 )
#X_train_r = chi2_selector.fit_transform( X_train, y_train )
#X_test_r = chi2_selector.transform( X_test )


# using Genetic algorithm
pop = ga.Population(int(0.2*X_train.shape[1]), X_train, X_test, y_train, y_test, pop_size=1000, 
                    models=[MultinomialNB(alpha=9.05)], model_weigths=[1])

for i in range(5000):
    pop.make_next_gen()
    features = pop.get_best_chromosome()
    print(max(pop.fitnesses))

features = pop.get_best_chromosome()
X_train_r = X_train[:, features]
X_test_r = X_test[:, features]


# Evaluation of the quality of feature selection using random Forest and
# decision tree algorithm

MODELS = [MultinomialNB(alpha = 9.05), RandomForestClassifier(), SGDClassifier()]

#MODELS = [MultinomialNB(), RandomForestClassifier(), DecisionTreeClassifier(),
#          SGDClassifier(max_iter=5, tol=-np.infty)]

for model in MODELS:
    name = str(type(model)).split(".")[-1]
    score_full = model.fit(X_train, y_train).score(X_test, y_test)
    score_reduced = model.fit(X_train_r, y_train).score(X_test_r, y_test)
    print("\n", name, "\n\t", "reduced score: ", "\t" + str(score_reduced),
          "\n\t", "full score: ", "\t" + str(score_full))
    print("-----------")
