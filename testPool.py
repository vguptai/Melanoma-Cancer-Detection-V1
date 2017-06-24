from multiprocessing import Pool as ThreadPool
from multiprocessing import Value, Lock
import itertools
from functools import partial
from ctypes import c_int

counter = Value(c_int)  # defaults to 0
counter_lock = Lock()

def loadAndSquashImage(imagePath):
    with counter_lock:
        counter.value += 1
    return imagePath

def testPoolOrder():
    imagesDataList = []
    imagePaths = []
    imagePaths.append('a')
    imagePaths.append('b')
    imagePaths.append('c')
    imagePaths.append('d')
    imagePaths.append('i')
    imagePaths.append('j')
    imagePaths.append('e')
    imagePaths.append('f')
    imagePaths.append('g')
    imagePaths.append('h')
    counter.value=0 # defaults to 0
    pool = ThreadPool(6)
    #imagesDataList = pool.map(loadAndSquashImage,itertools.izip(imagePaths,itertools.repeat(sizeX),itertools.repeat(sizeY)))
    imagesDataList = pool.map(partial(loadAndSquashImage),imagePaths)
    pool.close()
    pool.join()
    print imagesDataList

testPoolOrder()
