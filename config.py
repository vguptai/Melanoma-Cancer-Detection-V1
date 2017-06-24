n_classes = 2
numEpochs = 2000
numChannels = 3

batch_size = 32
testTrainSplit = 0.8
imageSizeX=224
imageSizeY=224
training_keep_rate = 0.5
testing_keep_rate = 1.0

#Data augmentation
oversample_minority = False
oversampling_multiplier = 10

#Batch Normalization
enableBatchNormalization = True

#Local Response Normalization
enableLocalResponseNormalization = False

#image standardization
enableImageStandardization = True

ckpt_dir = "./model"
logs_dir = "./logs"
#The folder where the dataset resides.
datasetFolder = "melanoma-dataset"

#Seeds to enable reproducible results
tensorflowSeed = 1234
randomSeed = 1234
numpySeed = 1234
opsSeed = 1234
dropoutSeed = 1234

#learning rate
learningRateInitial = 0.1
learningRateDecayFactor = 0.01
numEpochsPerDecay = 30
