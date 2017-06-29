from genericDataSetLoader import *
from config import *

genericDataSetLoader = genericDataSetLoader(False,datasetFolder,2,0.8,224,224)
#genericDataSetLoader.loadData()
genericDataSetLoader.analyzeDataDistribution()
#genericDataSetLoader.standardizeImages()
