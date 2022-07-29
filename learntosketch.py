import torch
import torch.nn as nn
import numpy as np
from hashes import *

class SketchNetwork(nn.Module):
    def __init__(self, K, R, d, OUT, aggregation, dropout_rate, hash_func, scale = 1.0):
        super(SketchNetwork, self).__init__()
        self.K = K
        self.R = R
        self.OUT = OUT
        self.d = d
        self.num_cel = 2**K
        self.scale = scale
        self.hash_func = hash_func
        if self.hash_func == 'SRP':
            self.h = SRP(self.K, self.R, self.d, self.OUT)
        elif self.hash_func == 'P-stable':
            self.h = PstableHash(self.R, self.d, self.OUT, self.scale)
        self.aggregation = aggregation
        self.sketch = nn.Parameter(torch.Tensor(OUT, R, self.num_cel),requires_grad = True)
        nn.init.normal_( self.sketch, 0.0, 0.0001 )
        self.dropout_rate = dropout_rate
        self.dropout= None
        if(dropout_rate > 0.0):
            self.dropout = nn.Dropout(p = self.dropout_rate)
            
        if(self.aggregation == "linear"):
            self.agg = nn.Parameter( torch.Tensor(R), requires_grad = True)
            self.agg.data.fill_(1.0)

    def init_param(self, race, hashes):
        self.h.data = hashes
        self.sketch.data = race

    def forward(self, X):
        with torch.no_grad():
            # X = torch.reshape(X, (X.shape[0], 28 * 28))
            hashcode = self.h.hash(X)
        if self.hash_func == 'SRP':
            if( self.dropout != None):
                input = self.dropout(self.sketch) * (1.0 - self.dropout_rate)
                alphas = torch.gather( input = input , dim = 2, index = hashcode)
            else:
                # print(self.sketch.shape)
                # print(hashcode.shape)
                alphas = torch.gather( input = self.sketch , dim = 2, index = hashcode)
        elif self.hash_func == 'P-stable':
            hashcode[hashcode < 0] = 0
            if( self.dropout != None):
                input = self.dropout(self.sketch) * (1.0 - self.dropout_rate)
                alphas = torch.gather( input = input , dim = 2, index = hashcode)
            else:
                # print(self.sketch.shape)
                # print(hashcode.shape)
                alphas = torch.gather( input = self.sketch , dim = 2, index = hashcode)
        alphas = alphas.permute(2,0,1) # alphas [OUT, R, B] -> [B, OUT, R]

        if(self.aggregation == "avg"):
            predict = torch.mean(alphas, dim = 2)
        elif(self.aggregation == "linear"):
            alphas = alphas * self.agg
            predict = torch.mean(alphas, dim = 2)
        return predict

    
    def get_memory(self):
        agg_cost = 0
        if(self.aggregation == "linear"):
            agg_cost = self.OUT * self.R
        return self.h.get_memory() + (self.OUT * self.R * self.num_cel + agg_cost) * 32
    
    def get_flops(self):
        return self.h.get_flops() + self.R * self.OUT
