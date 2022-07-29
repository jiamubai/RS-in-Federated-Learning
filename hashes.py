import torch
import torch.nn as nn
import numpy as np
import random


"""
Sign Random Projection
"""
class SRP(nn.Module):
    def __init__(self, K, R, d, OUT):
        super(SRP, self).__init__()
        self.K = K
        self.R = R
        self.OUT = OUT
        self.d = d
        self.num_cel = 2**K

        self.h = nn.Parameter( torch.Tensor(OUT, K*R, d), requires_grad = False)
        self.init_hashes()

        powersOfTwo = np.array([2**i for i in range(self.K)])
        self.powersOfTwo = torch.from_numpy(powersOfTwo).float()
        self.register_buffer('powersOfTwo_c', self.powersOfTwo)

    def weighted_values(self, values, probabilities, size):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(np.random.random_sample(size), bins)]                                                                     

    def generateSparseSRP(self, N, d):
        _v = np.array([1, -1, 0])
        self._prob = np.array([0.1667,0.1667, 0.6666])
        #self._prob = np.array([0.05,0.05, 0.9])
        #self._prob = np.array([0.01,0.01, 0.98])
        return self.weighted_values(_v, self._prob, (d*N)).reshape(N, d)

    def generateDenseSRP(self, N, d):
        return np.random.normal(size = (N,d))
    
    def init_hashes(self):
        sparse = self.generateSparseSRP(self.K*self.R, self.d)
        # print('sparse:', sparse.shape)
        hashes_init = []
        for _ in range(self.OUT):
            hashes_init.extend(torch.unsqueeze(
                               torch.from_numpy(sparse), dim = 0))
            # print(i, np.array(hashes_init).shape)
        # hashes_init = []
        # for _ in range(self.OUT):
        #     hashes_init.extend(torch.unsqueeze(
        #                        torch.from_numpy(self.generateSparseSRP(self.K*self.R, self.d)), dim = 0))
            # hashes_init.extend(torch.unsqueeze(
            #                     torch.from_numpy(self.generateDenseSRP(self.K*self.R, self.d)), dim = 0))
        hashes_init = torch.stack(hashes_init) 
        self.h.data = hashes_init.float()
        # print('self.h:', self.h.shape)
    
    def hash(self, X):
        with torch.no_grad():
            # print(self.h.shape)
            # print(X.shape)
            hashcode = self.h.matmul(X.permute(1, 0)).permute(2, 0 ,1)   #[OUT, K*R, B] -> [B, OUT, K*R]
            hashcode = torch.stack(torch.chunk(hashcode, self.R, dim = 2)).permute(1, 2, 0, 3) #[B, OUT, R, K]

            hashcode = torch.sign(hashcode)
            hashcode = torch.gt(hashcode, 0).float()

            hashcode = torch.matmul(hashcode, self.powersOfTwo_c).long()  #[B, OUT, R], hashcode same for each class
            hashcode = hashcode.permute(1,2,0)   #[B, OUT, R] -> [OUT, R, B]
            # print(hashcode.shape)
        
        return hashcode
    
    def get_flops(self):
        return self.d * self.K * self.R * (1 - self._prob[2])
    
    def get_memory(self):
        return 1
        return self.OUT * self.K * self.R * self.d

"""
P-stable Projection
"""

class PstableHash(nn.Module):
    def __init__(self, R, d, OUT, scale, p=2.0):
        super(PstableHash, self).__init__()
        # R: num of random hash functions
        self.R = R
        self.OUT = OUT
        # d: input dimension
        self.d = d
        self.scale = scale
        # self.num_cel = 2 ** K
        self.p = p

        self.h = nn.Parameter(torch.Tensor(OUT, R, d), requires_grad=False)
        self.init_hashes()
        self.b = nn.Parameter(torch.Tensor(OUT, R), requires_grad=False)
        self.init_bias()

        # powersOfTwo = np.array([2 ** i for i in range(self.K)])
        # self.powersOfTwo = torch.from_numpy(powersOfTwo).float()
        # self.register_buffer('powersOfTwo_c', self.powersOfTwo)

    def init_proj(self):
        if self.p == 2.0:
            return nn.Parameter(torch.abs(torch.normal(0, 1, size=(self.d, self.R))))

    def init_bias(self):
        max = self.scale
        min = 0.0
        # create tensor with random values in range (min, max)
        self.b.data = (max - min) * torch.rand((self.OUT, self.R)) + min
        # rand_tensor = torch.full((self.d, self.R), 1.0)
        # return nn.Parameter(rand_tensor)

    def generateSparse(self, N, d):
        _v = np.array([1, -1, 0])
        self._prob = np.array([0.1667, 0.1667, 0.6666])
        # self._prob = np.array([0.05,0.05, 0.9])
        # self._prob = np.array([0.01,0.01, 0.98])
        return self.weighted_values(_v, self._prob, (d * N)).reshape(N, d)

    def init_hashes(self):
        # sparse = self.generateSparse(self.R, self.d)
        # print('torch normal:', torch.normal(0, 1, size=(self.R, self.d)).shape)
        hashes_init = []
        for i in range(self.OUT):
            hashes_init.extend(torch.unsqueeze(torch.normal(0, 1, size=(self.R, self.d)), dim=0))
            # hashes_init.extend(torch.unsqueeze(torch.full((self.R, self.d), 1.0), dim=0))

        hashes_init = torch.stack(hashes_init)
        self.h.data = hashes_init.float()
        # print('self.h:', self.h.shape)

    def hash(self, X):
        with torch.no_grad():
            # print(self.h.shape)

            hashcode = self.h.matmul(X.permute(1, 0))
            # print('hashcode:', hashcode.shape)
            hashcode = hashcode.permute(2, 0, 1)  # [OUT, K*R, B] -> [B, OUT, K*R]

            for i in range(hashcode.shape[0]):
                # print('hashcode:', hashcode.shape)
                # print('hashcode 0 :', hashcode[i].shape)
                hashcode[i] = torch.add(hashcode[i], self.b)/self.scale

            # hashcode = torch.stack(torch.chunk(hashcode, self.R, dim=2)).permute(1, 2, 0, 3)  # [B, OUT, R, K]
            # print(hashcode.shape)
            # print(self.b.shape)

            # rand_tensor = (self.scale - 0.0) * torch.rand(hashcode.shape) + 0.0
            # rand_tensor = rand_tensor.to('cuda:2')
            # print(rand_tensor.shape)
            # hashcode = (hashcode + rand_tensor) / self.scale
            # hashcode = torch.round(hashcode)

            # hashcode = torch.matmul(hashcode, self.powersOfTwo_c).long()  # [B, OUT, R], hashcode same for each class
            hashcode = hashcode.permute(1, 2, 0)  # [B, OUT, R] -> [OUT, R, B]
            # hashcode = torch.abs(hashcode)
            hashcode = hashcode.long()
            # print(hashcode[0,0,:20])
        return hashcode

