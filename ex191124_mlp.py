import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, Din, H1, H2, Dout):
        super(MLP, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(Din, H1, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(self.hidden1[0].out_features, H2, bias=True),
            nn.ReLU()
        )
        self.softmax = nn.Sequential(
            nn.Linear(self.hidden2[0].out_features, Dout, bias=True),
            nn.LogSoftmax()
        )

    def forward(self, X):
        X = self.hidden1(X)
        X = self.hidden2(X)
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
            #print('@@', sqe)
            labEstimated = torch.argmax(output, dim=1)
            ncorrect += (Y[ii] == labEstimated).sum().item()            

    return loss, ncorrect