random.seed(randomSeed)
tf.set_random_seed(tensorflowSeed)
np.random.seed(numpySeed)

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
