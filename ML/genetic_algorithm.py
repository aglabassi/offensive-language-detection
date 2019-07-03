#!/usr/bin/env python3
#Contains classes useful for performing GA for features selection 
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:49:02 2019

@author: Abdel Ghani Labassi
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  SGDClassifier
import numpy as np
import random as rd

# Individual are defined by it chromosome, that is, a subset of fixed size k of
# the complete vocabulary
class Individual:
    
    def __init__( self, chromosome ):
        self.chromosome = chromosome
        self.k = len(chromosome)
        
        
    # Return the fitness evaluation of current individual. Assumes dataset is balanced
    def fitness(self, models, model_weigths, X_train, X_test, y_train, y_test ):
        
        ch = self.chromosome
        
        # We only fit selected features, i.e self.chromosome
        accuracies = [ model.fit( X_train[:,ch], y_train ).score( X_test[:,ch], y_test ) for model in models ]
            
        return np.dot(accuracies, model_weigths)
    


# Population class. Contains the "population characteristic" that allows fitness calculation,
# i.e X_train, X_test, y_train and y_test.
class Population:
    
    # Initialyzse the population
    def __init__(self, n_components , X_train, X_test, y_train, y_test, pop_size=200, 
                 models=[MultinomialNB(), SGDClassifier(max_iter=5, tol=-np.infty)], model_weigths=[0.7, 0.3]):
        
        self.X_train, self.X_test = X_train, X_test   
        self.y_train, self.y_test = y_train, y_test
        self.models, self.model_weigths = models, model_weigths
        
        # Trivial feature extraction
        self.genes = [ i for i in range(X_test[0].shape[1])]
        
        # Generating the first generation 
        individuals = self._generate_firstgen(set(self.genes), pop_size, n_components)
                                             

        # Init individuals and their fitnesses  
        self.individuals, self.fitnesses,  = [], []
        self._update(individuals)
    

    # Creating individuals with replacement of genes
    def _generate_firstgen(self,available_genes, pop_size, chr_size):
       
        individuals = []
        
        for i in range(pop_size):
            chromosome = np.array( rd.sample( available_genes, chr_size ) )
            individuals.append( Individual(chromosome) )
        
        return individuals
    
     
    # Takes an iterable of individuals and adds it to population by updating
    # self.fitnesses and self.individuals
    def _update(self, new_individuals):
        
        new_fitnesses = [ individual.fitness(self.models, self.model_weigths, self.X_train, self.X_test,self.y_train, self.y_test) 
                                for individual in new_individuals ]
        
        self.individuals = np.concatenate((self.individuals, new_individuals))
        self.fitnesses = np.concatenate((self.fitnesses, new_fitnesses))
            
            
    
    #Take 2 individuals in iterable and return their two children.
    def _reproduce(self, parents_idx, pc = 0.9, pm = 0.05):
        
        parents = [self.individuals[idx] for idx in parents_idx]
        
        # min_parent is parent who has lower chromosome size. At first, there's
        # only one individual that has lower chromosome size, which is due to 
        # genes attribution in first generation
        min_parent = 1 if parents[0].k > parents[1].k else 0   
        min_chr_size = parents[min_parent].k
        
        # Crossover
        if rd.random() < pc:
            
            #Cutting points, we use 2-points crossover for better results
            t1 = rd.randint(0,min_chr_size)
            t2 = rd.randint(t1,min_chr_size)
            t3 = rd.randint(t2,min_chr_size)
            
            chromo1 = np.concatenate((parents[0].chromosome[:t1],    
                                      parents[1].chromosome[t1:t2],
                                      parents[0].chromosome[t2:t3],
                                      parents[1].chromosome[t3:],  
                                      parents[not min_parent].chromosome[min_chr_size:] ))
            
            chromo2 = np.concatenate((parents[1].chromosome[:t1], 
                                      parents[0].chromosome[t1:t2],
                                      parents[1].chromosome[t2:t3],
                                      parents[0].chromosome[t3:],
                                      parents[not min_parent].chromosome[min_chr_size:] ))
            
            chromosomes = [ chromo1, chromo2 ]
            
        # no crossover
        else:
            chromosomes = [ parents[0].chromosome, parents[1].chromosome ]
        
        
        # Mutations: for each child's gene mutate whith probability pm
        for i in range(2):
            for j in range(min_chr_size): 
                if rd.random() < pm:
                    chromosomes[i][j] = rd.choice( self.genes )
                    
        return [ Individual(chromosomes[0]), Individual(chromosomes[1]) ] 
    
    
    # Outputs a mating pool of population size, using ranking selection, i.e weigths
    # are the ranks of the individuals.
    def _geta_mating_pool(self):
        
        mating_pool = rd.choices([ j for j in range(len(self)) ], self._get_ranks(), k=len(self))
        
        #Uniform shuffling is important for later-on uniform matching
        np.random.shuffle(mating_pool)
        
        return mating_pool
            
        
    
    # Produces next generation by replacing the current one
    def make_next_gen(self):
        
        mating_pool = self._geta_mating_pool()
        
        #Acouplement of consecutive pairs in the mating_pool
        new_individuals = []
        
        while mating_pool:  
            parents_idxs = [ mating_pool.pop() ,  mating_pool.pop() ]
            childrens = self._reproduce( parents_idxs ) 
            new_individuals =  new_individuals + [ childrens[0], childrens[1] ]
        
        self._update( new_individuals )
            
        #Delete n worst individuals by keeping the n bests
        w = self._get_ranks()
        normalized_w = w/(np.sum(w))
        good_idxs = np.random.choice([ i for i in range(len(self)) ], len(self)//2, replace = False, p = normalized_w) 
        
        #Update the attributes
        self.individuals = self.individuals[ good_idxs ]
        self.fitnesses = self.fitnesses[ good_idxs ]
            
    
        
    # Returns an array giving the ranks of the individuals   
    def _get_ranks(self):
        
        sorted_ind_idxs = np.argsort(self.fitnesses)
        
        res = np.zeros(len(self))
        for rank, ind_idx in enumerate(sorted_ind_idxs):
            res[ind_idx] = rank
            
        return res
        
        
    
    def get_best_chromosome(self):
        idx_max = np.where( self.fitnesses == max( self.fitnesses ))[0][0]     
        return self.individuals[idx_max].chromosome
    
    
    # Iterates in random order
    def __iter__(self):
        for individual in self.individuals:
            yield individual
            
    def __len__(self):
        return len(self.individuals)
