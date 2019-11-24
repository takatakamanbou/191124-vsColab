import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):

    def __init__(self, in_shape, C1, C2, H, Dout):

        super(CNN, self).__init__()

        X = torch.rand((1,) + in_shape)
        print('# input:', X.shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, C1, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        X = self.conv1(X)
        print('# conv1 output:', X.shape)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv1[0].out_channels, C2, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        X = self.conv2(X)
        print('# conv2 output:', X.shape)

        self.flatten = nn.Flatten()
        X = self.flatten(X)
        print('# flatten output:', X.shape)
        
        self.fc = nn.Sequential(
            nn.Linear(X.shape[1], H, bias=True),
            nn.ReLU()
        )
        X = self.fc(X)
        print('# fc output:', X.shape)

        self.softmax = nn.Sequential(
            nn.Linear(self.fc[0].out_features, Dout, bias=True),
            nn.LogSoftmax()
        )
        X = self.softmax(X)
        print('# softmax output:', X.shape)


    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.flatten(X)
        X = self.fc(X)
        X = self.softmax(X)

        return X


def evaluate(model, X, Y, bindex):

    nbatch = bindex.shape[0]
    loss = 0.0
    ncorrect = 0
    with torch.no_grad():
        for ib in range(nbatch):
            ii = np.where(bindex[ib, :])[0]
            output = model(X[ii, ::])
            #print('@', output.shape)
            loss += F.nll_loss(output, Y[ii], reduction='sum').item()
            labEstimated = torch.argmax(output, dim=1)
            ncorrect += (Y[ii] == labEstimated).sum().item()            

    return loss, ncorrect