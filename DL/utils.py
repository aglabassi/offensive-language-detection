#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:17:14 2019

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:57:03 2019

@author: Abdel
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Transform texts in arrays of the tokens' indexs in vocab
def raw_texts_to_sequences(texts, tokenizer, vocab):
        
        tkzd_texts = [ tokenizer(text) for text in texts ]
        
        def get_idxs(tkzd_text):
            sequence = []
            for word in tkzd_text:
                try:
                    sequence.append(vocab[word].index)
                except:
                    ''                
            return sequence
        
        #Getting the sequences
        sequences = [ get_idxs(tkzd_text) for tkzd_text in tkzd_texts ]
           
        #Padding for getting a square matrix
        maxlen = max([len(x) for x in sequences])
        X = pad_sequences(sequences, padding='post', maxlen=maxlen)
        
        return X
                    


def narray_to_tensor_dataset(X_train, X_test, y_train, y_test):
        
    inputs_train, targets_train = torch.tensor(X_train, dtype = torch.long), torch.tensor(np.array(y_train), dtype = torch.float32 ) 
    inputs_test, targets_test = torch.tensor(X_test, dtype = torch.long), torch.tensor(np.array(y_test), dtype = torch.float32 ) 
    
    return TensorDataset(inputs_train, targets_train), TensorDataset(inputs_test, targets_test)
    

def train_model(net, trainloader, criterion, optimizer, metric_calculator, validation_set):
    
    net.train()
    epoch_loss = 0
    epoch_mt = 0
    
    for i, data in enumerate(trainloader, 0):
        #Computing predictions
        inputs, targets = data
        outputs = net(inputs)
        predictions = torch.round(outputs.detach().squeeze())
        
        #Updating parameters
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        #metric calculation for training set
        metric = metric_calculator(targets, predictions)
        epoch_loss += loss.item()
        epoch_mt += metric.item()
        

    #Normalizing metrics for training set
    epoch_loss = epoch_loss / len(trainloader) 
    epoch_mt = epoch_mt / len(trainloader)
    
    #Getting predictions for validation set
    inputs = torch.tensor([ list(instance[0]) for instance in validation_set ])
    targets = torch.tensor([ instance[1] for instance in validation_set ])
    outputs = net(inputs)
    predictions = torch.round(outputs.detach().squeeze())
    
    #Computing metrics for validation set
    epoch_loss_valid = criterion(outputs.squeeze(), targets)
    epoch_mt_valid = metric_calculator(targets,predictions)
    
    return epoch_loss, epoch_mt, epoch_loss_valid, epoch_mt_valid

