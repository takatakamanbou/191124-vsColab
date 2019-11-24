import numpy as np
import mnist

### making mini batch indicies
#
def makeBatchIndex(N, batchsize):

    idx = np.random.permutation(N)
        
    nbatch = int(np.ceil(float(N) / batchsize))
    idxB = np.zeros(( nbatch, N ), dtype = bool)
    for ib in range(nbatch - 1):
        idxB[ib, idx[ib*batchsize:(ib+1)*batchsize]] = True
    ib = nbatch - 1
    idxB[ib, idx[ib*batchsize:]] = True

    return idxB


### loading the data
#
def loadMNISTData(pathMNIST='./', forCNN=False):

    mn = mnist.MNIST(pathMNIST=pathMNIST)
    datL = mn.getImage('L')
    if forCNN:
        datL = datL.reshape((-1, 1, 28, 28))
    labL = mn.getLabel('L')
    print(datL.shape, labL.shape)
    datT = mn.getImage('T')
    if forCNN:
        datT = datT.reshape((-1, 1, 28, 28))
    labT = mn.getLabel('T')
    print(datT.shape, labT.shape)
    
    return datL, labL, datT, labT
