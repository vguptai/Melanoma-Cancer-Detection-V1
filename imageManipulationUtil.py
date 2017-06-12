from PIL import Image
from numpy import array
import numpy as np
import scipy.misc
#from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool as ThreadPool
from multiprocessing import Value, Lock
import itertools
from functools import partial
from ctypes import c_int

numParallelProcess = 16
counter = Value(c_int)  # defaults to 0
counter_lock = Lock()

def squashImageArray(imageDataArray,sizeX,sizeY):
    return scipy.misc.imresize(imageDataArray,(sizeX,sizeY))

def loadImageAsArray(imagePath):
    imageData = Image.open(imagePath)
    imageData.load()
    return np.asarray(imageData)

def testSquashing():
    imageData = loadImageAsArray("test.jpg")
    imageDataScaled = squashImageArray(imageData,224,224)
    img1 = Image.fromarray(imageDataScaled, 'RGB')
    img1.show()
    img2 = Image.fromarray(imageData, 'RGB')
    img2.show()

def loadAndSquashImage(imagePath,sizeX,sizeY,totalImages):
    with counter_lock:
        counter.value += 1
        print "Image Processed:"+str(counter.value)+"/"+str(totalImages)
    return squashImageArray(loadImageAsArray(imagePath),sizeX,sizeY)

# def loadAndSquashImagesParallely(imagePaths,sizeX,sizeY):
#     imageXSize = sizeX
#     imageYSize = sizeY
#     imagesDataList = []
#     count = 0
#     with ProcessPoolExecutor(max_workers=numParallelProcess) as executor:
#         for imageNpArray in executor.map(loadAndSquashImage, imagePaths):
#             imagesDataList.append(imageNpArray)
#     return imagesDataList

#http://eli.thegreenplace.net/2013/01/16/python-paralellizing-cpu-bound-tasks-with-concurrent-futures/
#http://stackoverflow.com/questions/2846653/how-to-use-threading-in-python
def loadAndSquashImagesParallely(imagePaths,sizeX,sizeY):
    imagesDataList = []
    counter.value=0 # defaults to 0
    pool = ThreadPool(numParallelProcess)
    #imagesDataList = pool.map(loadAndSquashImage,itertools.izip(imagePaths,itertools.repeat(sizeX),itertools.repeat(sizeY)))
    imagesDataList = pool.map(partial(loadAndSquashImage,sizeX=sizeX,sizeY=sizeY,totalImages=len(imagePaths)),imagePaths)
    pool.close()
    pool.join()
    return np.array(imagesDataList)
