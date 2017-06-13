# Melanoma-Cancer-Detection-V1

Melanoma Cancer is a dangerous form of skin-cancer. Though, it is a rare form of skin-cancer but it is highly fatal. In this repo, we are training a Deep CNN for finding out if a lesion is cancerous or not.

The (https://medium.com/@vikram.gupta/detection-of-melanoma-cancer-using-deep-learning-85a7878e15eb)[medium article] explains the approach and the results in detail.

<b>Steps to train and test the model</b>

1) Delete all the checkpoints from the "model" directory before training a new model from scratch.

2) Run the following command. This will read all the images from the dataset folder and split them into training and testing set and pickle them. This is done to avoid loading the images multiple times, so skip this step if you have already done this before.
```python
python prepareDataSetFromImages.py 
```

3) Run the following command to train the models. 
```python
python convNetTrain.py
```

4) Run the following command to test the model.
```python
python convNetTest.py 
```

The file <b> config.py </b> contains the various parameters/flags that can be set.
