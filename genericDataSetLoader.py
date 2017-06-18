from os import walk
from os import path
from random import shuffle
import math
import tensorflow as tf
from PIL import Image
from numpy import array
import numpy as np
import scipy.misc
import imageManipulationUtil
import dataManipulationUtil
import pickle
import util
from config import *

tf.set_random_seed(tensorflowSeed)
np.random.seed(numpySeed)

class genericDataSetLoader:

    basePath = "dataset"
    numClasses = 2
    numChannels = 3
    imageXSize = 224
    imageYSize = 224
    splitPercentage = 0.8
    filePaths = []
    totalDataY = []
    className2ClassIndxMap = {}
    trainingDataX = None
    testingDataX = None
    trainingDataY = None
    testingDataY = None
    trainingDataOffset = 0
    testingDataOffset = 0
    alreadySplitInTrainTest = False

    def __init__(self,alreadySplitInTrainTest,basePath,numClasses,splitPercentage,imageSizeX,imageSizeY):
        self.basePath = basePath
        self.numClasses = numClasses
        self.splitPercentage = splitPercentage
        self.imageXSize = imageSizeX
        self.imageYSize = imageSizeY
        self.alreadySplitInTrainTest = alreadySplitInTrainTest

    def __initializeClass2IndxMap(self,classList):
        idx=0
        for clazz in classList:
            self.className2ClassIndxMap[clazz]=idx
            idx=idx+1
        print self.className2ClassIndxMap

    def __getClassIndex(self,clazz):
        return self.className2ClassIndxMap[clazz]

    def __loadImageDataParallely(self,fileNames):
        imagesDataList = imageManipulationUtil.loadAndSquashImagesParallely(fileNames,self.imageXSize,self.imageYSize)
        return imagesDataList

    def __loadAnImageData(fileName):
        imageXSize=64
        imageYSize=64
        return imageManipulationUtil.loadAndSquash(fileName,imageXSize,imageYSize)

    def __loadImageData(self,fileNames):
        imagesDataList = []
        cnt=0
        totalCnt = len(fileNames)
        for fName in fileNames:
            cnt = cnt+1
            print "Loading Image:"+str(cnt)+"/"+str(totalCnt)
            imagesDataList.append(self.__loadAnImageData(fName))
        img_np = np.array(imagesDataList)
        return img_np

    def __convertLabelsToOneHotVector(self,labelsList,numClasses):
        labelsArray = np.array(labelsList)
        oneHotVector = np.zeros((labelsArray.shape[0],numClasses))
        oneHotVector[np.arange(labelsArray.shape[0]), labelsArray] = 1
        return oneHotVector

    def __shuffle(self,list1,list2):
        list1_shuf = []
        list2_shuf = []
        index_shuf = range(len(list1))
        shuffle(index_shuf)
        for i in index_shuf:
            list1_shuf.append(list1[i])
            list2_shuf.append(list2[i])
        return list1_shuf,list2_shuf

    def __trainTestSplit(self,filePaths,labels,splitPercentage):
        splitIndex = int(math.ceil(splitPercentage*len(filePaths)))
        trainingDataX = filePaths[:splitIndex]
        trainingDataY = labels[:splitIndex]
        testingDataX = filePaths[splitIndex:]
        testingDataY = labels[splitIndex:]
        return trainingDataX,trainingDataY,testingDataX,testingDataY

    def prepareDataSetFromImages(self):
        if(self.alreadySplitInTrainTest):
            print "Data is already split into training,testing.Loading data..."
            self.__prepareDataSetFromAlreadySplitImages()
        else:
            print "Loading data after splitting and shuffling..."
            self.__prepareDataSetFromImagesSplitShuffle()

    def __prepareDataSetFromAlreadySplitImages(self):
        trainTestDirectory = next(walk(self.basePath))[1]
        if(len(trainTestDirectory)!=2):
            raise Exception("Number of split found more than 2. Expect only Train/Test")
        trainingDirs = next(walk(self.basePath+"/training"))[1]
        testingDirs = next(walk(self.basePath+"/testing"))[1]
        #print "Training directories:"+(str(trainingDirs))
        #print "Testing directories:"+(str(testingDirs))
        self.__initializeClass2IndxMap(trainingDirs)
        self.__initializeClass2IndxMap(testingDirs)
        self.trainingDataX = []
        self.trainingDataY = []
        self.testingDataX = []
        self.testingDataY = []
        for trainingDir in trainingDirs:
            trainingClassFiles = next(walk(path.join(self.basePath+"/training",trainingDir)))[2]
            for fName in trainingClassFiles:
                self.trainingDataX.append(self.basePath+"/training"+"/"+trainingDir+"/"+fName)
                self.trainingDataY.append(self.__getClassIndex(trainingDir))

        print "Shuffling the training dataset..."
        self.trainingDataX,self.trainingDataY = self.__shuffle(self.trainingDataX,self.trainingDataY)

        for testingDir in testingDirs:
            testingClassFiles = next(walk(path.join(self.basePath+"/testing",testingDir)))[2]
            for fName in testingClassFiles:
                self.testingDataX.append(self.basePath+"/testing"+"/"+testingDir+"/"+fName)
                self.testingDataY.append(self.__getClassIndex(testingDir))

        print "Shuffling the testing dataset..."
        self.testingDataX,self.testingDataY = self.__shuffle(self.testingDataX,self.testingDataY)

        self.__postProcessData()
        print self.testingDataX.shape
        print self.testingDataY
        print self.trainingDataX.shape
        print self.trainingDataY
        self.__save()

    def __prepareDataSetFromImagesSplitShuffle(self):
        self.trainingDataOffset = 0
        classDirectories = next(walk(self.basePath))[1]
        if(len(classDirectories)!=self.numClasses):
            raise Exception("Number of classes found in dataset is not equal to the specified numClasses")
        self.__initializeClass2IndxMap(classDirectories)
        dataMap = {}
        self.trainingDataX = []
        self.trainingDataY = []
        self.testingDataX = []
        self.testingDataY = []
        for classDirectory in classDirectories:
            classFiles = next(walk(path.join(self.basePath,classDirectory)))[2]
            filePaths = []
            totalDataY = []
            dataMap[classDirectory] = {}
            for fname in classFiles:
                filePaths.append(self.basePath+"/"+classDirectory+"/"+fname)
                totalDataY.append(self.__getClassIndex(classDirectory))
            dataMap[classDirectory]["filePaths"] = filePaths
            dataMap[classDirectory]["fileLabels"] = totalDataY

        #split into train-test
        for key,value in dataMap.iteritems():
            dataMap[key]["trainingDataX"],dataMap[key]["trainingDataY"],dataMap[key]["testingDataX"],dataMap[key]["testingDataY"]  = self.__trainTestSplit(dataMap[key]["filePaths"],dataMap[key]["fileLabels"],self.splitPercentage)
            self.trainingDataX.extend(dataMap[key]["trainingDataX"])
            self.trainingDataY.extend(dataMap[key]["trainingDataY"])
            self.testingDataX.extend(dataMap[key]["testingDataX"])
            self.testingDataY.extend(dataMap[key]["testingDataY"])

        print "Shuffling the dataset..."
        #shuffle the dataset for randomization
        self.trainingDataX,self.trainingDataY = self.__shuffle(self.trainingDataX,self.trainingDataY)
        self.testingDataX,self.testingDataY = self.__shuffle(self.testingDataX,self.testingDataY)

        self.__postProcessData()
        self.__save()

    def __postProcessData(self):
        #convert file paths into numpy array by reading the files
        print "Reading the training image files..."+util.getCurrentTime()
        self.trainingDataX = self.__loadImageDataParallely(self.trainingDataX)
        print "Reading the training image files..."+util.getCurrentTime()
        self.testingDataX = self.__loadImageDataParallely(self.testingDataX)

        #convert class lables into one hot encoded
        print "Creating one hot encoded vectors for training labels..."+util.getCurrentTime()
        self.trainingDataY = self.__convertLabelsToOneHotVector(self.trainingDataY,self.numClasses)
        print "Creating one hot encoded vectors for testing labels..."+util.getCurrentTime()
        self.testingDataY = self.__convertLabelsToOneHotVector(self.testingDataY,self.numClasses)


    def loadData(self):
        pklFile = open("preparedData.pkl", 'rb')
        preparedData=pickle.load(pklFile)
        self.trainingDataX = preparedData["trainingX"]
        self.trainingDataY = preparedData["trainingY"]
        self.testingDataX = preparedData["testingX"]
        self.testingDataY = preparedData["testingY"]
        print "Data loaded..."
        print self.trainingDataX.shape
        print self.trainingDataY.shape
        print self.testingDataX.shape
        print self.testingDataY.shape

    def __save(self):
        print "Saving the processed data..."
        preparedData={}
        preparedData["trainingX"] = self.trainingDataX
        preparedData["trainingY"] = self.trainingDataY
        preparedData["testingX"] = self.testingDataX
        preparedData["testingY"] = self.testingDataY
        pklFile = open("preparedData.pkl", 'wb')
        pickle.dump(preparedData, pklFile)
        pklFile.close()
        print "Data saved..."

    def getNextTrainBatch(self,batchSize):
        trainDataX = dataManipulationUtil.selectRows(self.trainingDataX,self.trainingDataOffset,batchSize)
        trainDataY = dataManipulationUtil.selectRows(self.trainingDataY,self.trainingDataOffset,batchSize)
        self.trainingDataOffset = self.trainingDataOffset+batchSize
        return trainDataX,trainDataY

    def resetTrainBatch(self):
        self.trainingDataOffset=0

    def resetTestBatch(self):
        self.testingDataOffset=0

    def getNextTestBatch(self,batchSize):
        testDataX = dataManipulationUtil.selectRows(self.testingDataX,self.testingDataOffset,batchSize)
        testDataY = dataManipulationUtil.selectRows(self.testingDataY,self.testingDataOffset,batchSize)
        self.testingDataOffset = self.testingDataOffset+batchSize
        return testDataX,testDataY

    """
    Utility to analyze the distribution of data
    in training and testing set.
    """
    def analyzeDataDistribution(self):
        self.loadData()
        print "Total Training Instances:"+str(self.trainingDataY.shape[0])
        print "Total Testing Instances:"+str(self.testingDataY.shape[0])
        #print self.__convertOneHotVectorToLabels(self.trainingDataY)
        for classIndex in range(0,self.numClasses):
            print "Distribution For Class:"+str(classIndex)
            trainDistribution = self.__convertOneHotVectorToLabels(self.trainingDataY)
            trainDistribution = np.count_nonzero(trainDistribution == classIndex)
            testDistribution = self.__convertOneHotVectorToLabels(self.testingDataY)
            testDistribution = np.count_nonzero(testDistribution == classIndex)
            print "Instances In Training Data:"+str(trainDistribution)
            print "Instances In Testing Data:"+str(testDistribution)
        print "Done"

    def __convertOneHotVectorToLabels(self,oneHotVectors):
        labels = np.argmax(oneHotVectors==1,axis=1)
        return labels
