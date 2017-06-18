n_classes = 2
numEpochs = 2000
numChannels = 3

batch_size = 128
testTrainSplit = 0.8
imageSizeX=224
imageSizeY=224
training_keep_rate = 0.5
testing_keep_rate = 1.0

#Data augmentation
oversample_minority = True
oversampling_multiplier = 5

ckpt_dir = "./model"

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
learningRateDecayFactor = 0.16
numEpochsPerDecay = 30
