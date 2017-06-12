from genericDataSetLoader import *
from config import *

genericDataSetLoader = genericDataSetLoader(False,datasetFolder,n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.prepareDataSetFromImages()
