from random import shuffle
import math
import numpy as np
from config import *

np.random.seed(numpySeed)

def selectRows(dataArray,rowOffset,numOfRows):
    if(rowOffset>=dataArray.shape[0]):
        return None
    elif ((rowOffset+numOfRows)>dataArray.shape[0]):
        return dataArray[rowOffset:dataArray.shape[0],:]
    return dataArray[rowOffset:rowOffset+numOfRows,:]

def test():
    a = np.random.random((10,4))
    return selectRows(a,0,10)

#print(test())
