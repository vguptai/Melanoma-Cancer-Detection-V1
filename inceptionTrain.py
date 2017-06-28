import argparse
from constants import *
from InceptionV3 import *
import os.path
from tensorflow.python.framework import graph_util
import random
import numpy as np
from osUtils import *
import tensorflow as tf
import tarfile
from six.moves import urllib
import sys
import numpy as np
from PIL import Image
from config import *
from genericDataSetLoader import *

random.seed(randomSeed)
tf.set_random_seed(tensorflowSeed)
np.random.seed(numpySeed)

"""Download and extract model tar file.
If the pretrained model we're using doesn't already exist, this function
downloads it from the TensorFlow.org website and unpacks it into a directory.
"""
def download_and_extract_inception_model(modelDir):
  destDirectory = modelDir
  if not os.path.exists(destDirectory):
    os.makedirs(destDirectory)
  filename = INCEPTION_MODEL_URL.split('/')[-1]
  filepath = os.path.join(destDirectory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(INCEPTION_MODEL_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(destDirectory)

"""
Load the pretrained inception graph. Add FC layers as per our use case.
Also sets other parts of the model like optimizer, learning rate etc.
"""
def create_inception_graph(num_batches_per_epoch,FLAGS):
    modelFilePath = os.path.join(FLAGS.imagenet_inception_model_dir, INCEPTION_MODEL_GRAPH_DEF_FILE)
    inceptionV3 = InceptionV3(modelFilePath)
    inceptionV3.add_final_training_ops(FLAGS.num_classes,FLAGS.final_tensor_name,FLAGS.optimizer_name,num_batches_per_epoch, FLAGS)
    inceptionV3.add_evaluation_step()
    return inceptionV3

def setup_image_distortion_ops(inceptionV3,FLAGS):
    if DatasetManager.should_distort_images(FLAGS):
        inceptionV3.add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop,FLAGS.random_scale, FLAGS.random_brightness)

def calculateTrainAccuracy(sess):
    genericDataSetLoader.resetTrainBatch()
    batchAccuracies = []
    while(True):
        trainX, trainY = genericDataSetLoader.getNextTrainBatch(FLAGS.train_batch_size)
        if(trainX is None):
            break
        accuracy_batch, cross_entropy_value_batch = inceptionV3.evaluate(sess,trainX,trainY)
        batchAccuracies.append(accuracy_batch)
    print "Training Accuracy:"+ str(sum(batchAccuracies) / float(len(batchAccuracies)))

def calculateTestAccuracy(sess):
    genericDataSetLoader.resetTestBatch()
    batchAccuracies = []
    while(True):
        testX, testY = genericDataSetLoader.getNextTestBatch(FLAGS.test_batch_size)
        if(testX is None):
            break
        accuracy_batch, cross_entropy_value_batch = inceptionV3.evaluate(sess,testX,testY)
        batchAccuracies.append(accuracy_batch)
    print "Testing Accuracy:" +str(sum(batchAccuracies) / float(len(batchAccuracies)))

def restoreFromCheckPoint(sess,saver):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path+" :Restoring from a checkpoint...")
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        start = global_step.eval() # get last global_step
        start = start+1
    else:
        print "Starting fresh training..."
        start = global_step.eval() # get last global_step
    return start

def trainInceptionNeuralNetwork(inceptionV3,FLAGS):
    with tf.Session(graph=inceptionV3.inceptionGraph) as sess:
        sess.run(tf.global_variables_initializer())
        genericDataSetLoader.convertToBottleNecks(sess,FLAGS,inceptionV3)
        #Restore the model from a previous checkpoint if any and get the epoch from which to continue training
        #start = restoreFromCheckPoint(sess,saver)
        start = 0
        print "Start from:"+str(start)+"/"+str(numEpochs)
        prev_epoch_loss = 0
        #Training epochs
        for epoch in range(start,numEpochs):
            epoch_loss = 0
            genericDataSetLoader.resetTrainBatch()
            while(True):
                epoch_x, epoch_y = genericDataSetLoader.getNextTrainBatch(FLAGS.train_batch_size)
                if(epoch_x is None):
                    break
                _,c = inceptionV3.train_step(sess,epoch_x,epoch_y,FLAGS.dropout_keep_rate)
                epoch_loss += c

            if(prev_epoch_loss!=0):
                loss_improvement = (prev_epoch_loss - epoch_loss)/prev_epoch_loss
                if(loss_improvement<0.0):
                    print "Loss did not improved more than the threshold...quitting now.."+str(loss_improvement)
                    #break
                else:
                    print "Loss has improved more than the threshold...saving this model.."+str(loss_improvement)
            #saver.save(sess,'model/data-all.chkp',global_step=global_step)
            print "Epoch:"+str(epoch)+'/'+str(numEpochs)+" loss:" + str(epoch_loss)
            prev_epoch_loss = epoch_loss
            calculateTrainAccuracy(sess)
            #Get the validation/test accuracy
            calculateTestAccuracy(sess)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--output_graph',type=str,default='./tmp/output_graph', help='Where to save the trained graph.')
  parser.add_argument('--output_labels',type=str,default='./tmp/output_labels.txt',help='Where to save the trained graph\'s labels.')
  parser.add_argument('--summaries_dir',type=str,default='./tmp/retrain_logs',help='Where to save summary logs for TensorBoard.')
  parser.add_argument('--how_many_training_steps',type=int,default=500,help='How many training steps to run before ending.')
  parser.add_argument('--imagenet_inception_model_dir',type=str,default='./imagenetInception',help="""Path to classify_image_graph_def.pb,imagenet_synset_to_human_label_map.txt, and imagenet_2012_challenge_label_map_proto.pbtxt.""")
  parser.add_argument('--bottleneck_dir',type=str,default='./tmp/bottleneck',help='Path to cache bottleneck layer values as files.')
  parser.add_argument('--final_tensor_name',type=str,default='final_result',help="""The name of the output classification layer in the retrained graph.""")

  #Learning Rate and Optimizers
  parser.add_argument('--optimizer_name',type=str,default="rmsprop",help='Optimizer to be used: sgd,adam,rmsprop')
  parser.add_argument('--learning_rate_decay_factor',type=float,default=0.16,help='Learning rate decay factor.')
  parser.add_argument('--learning_rate',type=float,default=0.1,help='Initial learning rate.')
  parser.add_argument('--rmsprop_decay',type=float,default=0.9,help='Decay term for RMSProp.')
  parser.add_argument('--rmsprop_momentum',type=float,default=0.9,help='Momentum in RMSProp.')
  parser.add_argument('--rmsprop_epsilon',type=float,default=1.0,help='Epsilon term for RMSProp.')
  parser.add_argument('--num_epochs_per_decay',type=int,default=30,help='Epochs after which learning rate decays.')
  parser.add_argument('--learning_rate_type',type=str,default="exp_decay",help='exp_decay,const')

  #Normalizations/Regularizations
  parser.add_argument('--use_batch_normalization',type=bool,default=False,help='Control the use of batch normalization')
  parser.add_argument('--dropout_keep_rate',type=float,default=0.5)

  #Batch Sizes
  parser.add_argument('--train_batch_size',type=int,default=128,help='How many images to train on at a time.')
  parser.add_argument('--test_batch_size',type=int,default=128)

  parser.add_argument('--num_classes',type=int,default='2')

  #parse the parameters
  FLAGS, unparsed = parser.parse_known_args()

  #download the pretrained inceptionV3 model
  download_and_extract_inception_model(FLAGS.imagenet_inception_model_dir)

  #load the prepared dataset from the pickled file
  genericDataSetLoader = genericDataSetLoader()
  genericDataSetLoader.loadData()
  numTrainingBatches = genericDataSetLoader.numberOfTrainingBatches(FLAGS.train_batch_size)

  #load the pretrained inception graph and create the complete model
  inceptionV3 = create_inception_graph(numTrainingBatches, FLAGS)

  #train the neural network
  trainInceptionNeuralNetwork(inceptionV3,FLAGS)
