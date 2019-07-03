#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:56:33 2019

@author: Abdel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN4old(nn.Module):
    
    def __init__(self, vocab_size, emb_weights=None, nb_krnl=32, window_sizes=[1,2,3] , drop1=0.3, drop2=0.3):
        super(CNN4old, self).__init__()          
        
        embedding_dim = emb_weights.shape[1] if emb_weights is not None else 50
        
        try:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.embedding.weight.data.copy_(emb_weights)
        except:
            ''
            
        self.convs = nn.ModuleList([ nn.Conv2d(1, nb_krnl, (size,embedding_dim)) for size in window_sizes ])
        
        self.drop1 = nn.Dropout(drop1)
        
        self.drop2 = nn.Dropout(drop2)
        
        self.linear = nn.Linear(nb_krnl*len(self.convs),1)
        
    
    def forward(self,x):
        
        embedded = torch.unsqueeze(self.embedding(x) , 1)
        
        conveds= [F.relu(self.drop1(conv(embedded).squeeze(3))) for conv in self.convs]
                
        pooleds = [F.max_pool1d(conved, conved.shape[2]).squeeze(2) for conved in conveds]
        
        cated = torch.cat(pooleds, dim=1)
        
        cated = self.drop2(cated)
            
        return torch.sigmoid(self.linear(cated))
    
    
    
    def __str__(self):
        
        emb = str(self.embedding)
        convs = str(self.convs)
        drop1 = str(self.drop1)
        lin = str(self.linear)
        drop2 = str(self.drop2)
        
        return emb + "\n" + convs + "\n" + drop1 + "\nMaxPool \n" + drop2 + "\n" + lin + "\n"
        
        
        

#
#
#

