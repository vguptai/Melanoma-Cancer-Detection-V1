import tensorflow as tf
from genericDataSetLoader import *
from config import *
from convNetModel import *

genericDataSetLoader = genericDataSetLoader(False,"dataset",n_classes,testTrainSplit,imageSizeX,imageSizeY)
genericDataSetLoader.loadData()

tf.set_random_seed(tensorflowSeed)

def testNeuralNetwork():
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path+" :Testing this checkpoint...")
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        calculateTestAccuracy()

def calculateTestAccuracy():
    genericDataSetLoader.resetTestBatch()
    batchAccuracies = []
    while(True):
        testX, testY = genericDataSetLoader.getNextTestBatch(batch_size)
        if(testX is None):
            break
        acc = convNetModel.test(testX,testY)
        batchAccuracies.append(acc)
        print "Accuracy of test batch..."+str(acc)
    #testX = np.reshape(testX, (-1, imageSizeX, imageSizeY, numChannels))
    print('Accuracy:', sum(batchAccuracies) / float(len(batchAccuracies)))

def findWronglyLabelledExamples():
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path+" :Testing this checkpoint...")
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        genericDataSetLoader.resetTestBatch()
        class_1_wrong = 0
        class_1_correct = 0
        class_2_wrong = 0
        class_2_correct = 0
        i = 0
        while(True):
            i =i +1
            testX, testY = genericDataSetLoader.getNextTestBatch(1)
            if(testX is None):
                break
            class_index = np.argmax(testY==1,axis=1)+1
            acc = convNetModel.test(testX,testY)
            #print "Class index for example number "+str(i)+" "+str(class_index)
            if(acc==0):
                #print "Accuracy is zero"
                if class_index==1:
                    class_1_wrong = class_1_wrong + 1
                elif class_index==2:
                    class_2_wrong = class_2_wrong + 1
                else:
                    raise ValueError('Wrong class index')
            elif(acc==1):
                #print "Accuracy is one"
                if class_index==1:
                    class_1_correct = class_1_correct + 1
                elif class_index==2:
                    class_2_correct = class_2_correct + 1
                else:
                    raise ValueError('Wrong class index')
            else:
                raise ValueError('Something is wrong here')

        print "Error in class 1:"+str(class_1_wrong)
        print "Error in class 2:"+str(class_2_wrong)
        print "Correct labels in class 1:"+str(class_1_correct)
        print "Correct labels in class 2:"+str(class_2_correct)
        print "Class 1 Accuracy:" + str(float(class_1_correct)/(class_1_correct+class_1_wrong))
        print "Class 2 Accuracy:" + str(float(class_2_correct)/(class_2_correct+class_2_wrong))

convNetModel = convNetModel()
saver = tf.train.Saver()
#testNeuralNetwork()
findWronglyLabelledExamples()
