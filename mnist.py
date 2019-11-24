from __future__ import print_function

import struct
import os
import numpy as np
 
 
class MNIST:
 
    def __init__( self, pathMNIST = '.' ):
 
        fnLabelL = os.path.join( pathMNIST, 'train-labels-idx1-ubyte' )
        fnLabelT = os.path.join( pathMNIST, 't10k-labels-idx1-ubyte' )
        self.fnLabel = { 'L': fnLabelL, 'T': fnLabelT }
        fnImageL = os.path.join( pathMNIST, 'train-images-idx3-ubyte' )
        fnImageT = os.path.join( pathMNIST, 't10k-images-idx3-ubyte' )
        self.fnImage = { 'L': fnImageL, 'T': fnImageT }
        self.nrow = 28
        self.ncol = 28
        self.nclass = 10
 
 
    def getLabel( self, LT ):
        
        return _readLabel( self.fnLabel[LT] )
 
 
    def getImage( self, LT ):
        
        return _readImage( self.fnImage[LT] )
 
 
##### reading the label file
#
def _readLabel( fnLabel ):
 
    f = open( fnLabel, 'rb' )
 
    ### header (two 4B integers, magic number(2049) & number of items)
    #
    header = f.read( 8 )
    mn, num = struct.unpack( '>2i', header )  # MSB first (bigendian)
    assert mn == 2049
    #print mn, num
 
    ### labels (unsigned byte)
    #
    label = np.array( struct.unpack( '>%dB' % num, f.read() ), dtype = int )
 
    f.close()
 
    return label
 
 
##### reading the image file
#
def _readImage( fnImage ):
 
    f = open( fnImage, 'rb' )
 
    ### header (four 4B integers, magic number(2051), #images, #rows, and #cols
    #
    header = f.read( 16 )
    mn, num, nrow, ncol = struct.unpack( '>4i', header ) # MSB first (bigendian)
    assert mn == 2051
    #print mn, num, nrow, ncol
 
    ### pixels (unsigned byte)
    #
    npixel = ncol * nrow
    #pixel = np.empty( ( num, npixel ), dtype = int )
    #pixel = np.empty( ( num, npixel ), dtype = np.int32 )
    pixel = np.empty( ( num, npixel ) )
    for i in range( num ):
        buf = struct.unpack( '>%dB' % npixel, f.read( npixel ) )
        pixel[i, :] = np.asarray( buf )
 
    f.close()
 
    return pixel
 
 
 
if __name__ == '__main__':
 
    mnist = MNIST( pathMNIST = './mnist' )

    print( '# MNIST training data' )
    dat = mnist.getImage( 'L' )
    lab = mnist.getLabel( 'L' )
    print( dat.shape, dat.dtype, lab.shape )
 
    print( '# MNIST test data' )
    dat = mnist.getImage( 'T' )
    lab = mnist.getLabel( 'T' )
    print( dat.shape, dat.dtype, lab.shape )
    
