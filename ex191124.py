# -*- coding: utf-8 -*-
"""ex191124.ipynb

Automatically generated by Colaboratory. => modified by takataka

Original file is located at
    https://colab.research.google.com/drive/1YpOY0O2pbHbwzy2wcbhQrBVa5xgRU_qT

# ex1124

Google Colab と研究室のGPGPUマシンの性能比較のためのほげ

"""

"""## 初期設定"""

#from google.colab import drive
#drive.mount('gdrive')

#pathGDrive = 'gdrive/My Drive/191124-vsColab'
pathGDrive = '.'
#!ls 'gdrive/My Drive/191124-vsColab'

import numpy as np
import cv2
import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### GoogleDrive上の hoge.py を import できるように
#
#sys.path.append(pathGDrive)
import ex191124_data as data
#import ex191124_mlp as network
import ex191124_cnn as network

### device
#
use_gpu_if_available = True
if use_gpu_if_available and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('# using', device)

### loading and setting the data

#datL, labL, datT, labT = data.loadMNISTData(pathMNIST=os.path.join(pathGDrive, 'mnist'))
datL, labL, datT, labT = data.loadMNISTData(pathMNIST=os.path.join(pathGDrive, 'mnist'), forCNN=True)
NL = datL.shape[0]
NT = datT.shape[0]
K = 10
batchsize = 128

XL = torch.from_numpy(datL.astype(np.float32)).to(device)
YL = torch.from_numpy(labL).to(device)

bindexL = data.makeBatchIndex(NL, batchsize)
nbatchL = bindexL.shape[0]

print(datL.shape, XL.shape, YL.shape)

XT = torch.from_numpy(datT.astype(np.float32)).to(device)
YT = torch.from_numpy(labT).to(device)

bindexT = data.makeBatchIndex(NT, batchsize)
nbatchT = bindexT.shape[0]

print(datT.shape, XT.shape, YT.shape)

"""## 学習"""

# うっかり全てのセルを実行とかしたときのためのストッパー
#hoge

### initializing the network
#
torch.manual_seed(0)
#net = network.MLP(XL.shape[1], 1000, 1000, K)
net = network.CNN(XL.shape[1:], 16, 32, 1000, K)
model = net.to(device)
print(net)
#optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(optimizer)

### training
#
nitr = 10000
nd = 0
start = datetime.datetime.now()

for i in range(nitr):

    if (i < 500 and i % 100 == 0) or (i % 500 == 0):
        model.eval()
        loss, ncorrect = network.evaluate(model, XL, YL, bindexL)
        print(i, nd, loss/NL, ncorrect/NL)

    model.train()
    ib = np.random.randint(0, nbatchL)
    ii = np.where(bindexL[ib, :])[0]
    optimizer.zero_grad()
    output = model(XL[ii, ::])
    loss = F.nll_loss(output, YL[ii])
    loss.backward()
    optimizer.step()

    nd += ii.shape[0]

model.eval()
loss, ncorrect = network.evaluate(model, XL, YL, bindexL)
print(nitr, nd, loss/NL, ncorrect/NL)

print('# elapsed time: ', datetime.datetime.now() - start)

### saving the model
#
fnModel = pathGDrive + '/ex191124-params.pickle'
with open(fnModel, mode = 'wb') as f:
    torch.save(model.state_dict(), f)
    print('# The model is saved to ', fnModel)

"""## テスト"""

fnModel = pathGDrive + '/ex191124-params.pickle'

### loading the network
#
torch.manual_seed(0)
#net = network.MLP(XT.shape[1], 1000, 1000, K)
net = network.CNN(XL.shape[1:], 8, 8, 1000, K)
with open(fnModel, mode = 'rb') as f:
    net.load_state_dict(torch.load(f))
    model = net.to(device)
print(net)

model.eval()
loss, ncorrect = network.evaluate(model, XT, YT, bindexT)
print(loss/NT, ncorrect/NT)
