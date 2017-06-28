n_classes = 2
numEpochs = 2000
numChannels = 3

batch_size = 32
testTrainSplit = 0.8
imageSizeX=299
imageSizeY=299
training_keep_rate = 0.5
testing_keep_rate = 1.0

#Data augmentation
oversample_minority = False
oversampling_multiplier = 5

#Batch Normalization
enableBatchNormalization = False

#Local Response Normalization
enableLocalResponseNormalization = False

#image standardization
enableImageStandardization = False

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

#Inception Constants
INCEPTION_MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
INCEPTION_MODEL_GRAPH_DEF_FILE = 'classify_image_graph_def.pb'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
DECODED_JPEG_DATA_TENSOR_NAME = 'DecodeJpeg:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
FINAL_MINUS_1_LAYER_SIZE = 512
FINAL_MINUS_2_LAYER_SIZE = 512
