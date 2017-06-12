import tensorflow as tf
from genericDataSetLoader import *
from config import *
import os
from convNetModel import *

genericDataSetLoader = genericDataSetLoader(False,datasetFolder,n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.loadData()

def calculateTrainAccuracy():
    genericDataSetLoader.resetTrainBatch()
    batchAccuracies = []
    while(True):
        trainX, trainY = genericDataSetLoader.getNextTrainBatch(batch_size)
        if(trainX is None):
            break
        acc = convNetModel.test(trainX,trainY)
        batchAccuracies.append(acc)
        #print "Accuracy of test batch..."+str(acc)
    #testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
    print('Training Accuracy:', sum(batchAccuracies) / float(len(batchAccuracies)))

def calculateTestAccuracy():
    genericDataSetLoader.resetTestBatch()
    batchAccuracies = []
    while(True):
        testX, testY = genericDataSetLoader.getNextTestBatch(batch_size)
        if(testX is None):
            break
        acc = convNetModel.test(testX,testY)
        batchAccuracies.append(acc)
        #print "Accuracy of test batch..."+str(acc)
    #testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
    print('Testing/Validation Accuracy:', sum(batchAccuracies) / float(len(batchAccuracies)))


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

def trainNeuralNetwork():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #Restore the model from a previous checkpoint if any and get the epoch from which to continue training
        start = restoreFromCheckPoint(sess,saver)
        print "Start from:"+str(start)+"/"+str(numEpochs)

        prev_epoch_loss = 0
        #Training epochs
        for epoch in range(start,numEpochs):
            epoch_loss = 0

            genericDataSetLoader.resetTrainBatch()

            while(True):
                epoch_x, epoch_y = genericDataSetLoader.getNextTrainBatch(batch_size)
                if(epoch_x is None):
                    break
                _, c = convNetModel.train(sess,epoch_x,epoch_y)
                epoch_loss += c


            if(prev_epoch_loss!=0):
                loss_improvement = (prev_epoch_loss - epoch_loss)/prev_epoch_loss
                if(loss_improvement<0.01):
                    print "Loss did not improved more than the threshold...quitting now.."+str(loss_improvement)
                    break
                else:
                    print "Loss has improved more than the threshold...saving this model.."+str(loss_improvement)

            global_step.assign(epoch).eval()
            saver.save(sess,'model/data-all.chkp',global_step=global_step)
            print('Epoch', epoch, 'completed out of', numEpochs, 'loss:', epoch_loss)

            prev_epoch_loss = epoch_loss

            calculateTrainAccuracy()
            #Get the validation/test accuracy
            calculateTestAccuracy()

#if not os.path.exists(ckpt_dir):
#    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

convNetModel = convNetModel()
saver = tf.train.Saver()
trainNeuralNetwork()
