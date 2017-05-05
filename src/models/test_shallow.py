
import os

from keras.regularizers import l2

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from utils import *
from os.path import expanduser
from keras.applications import VGG16
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator
import numpy
import keras
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import Nadam
print "loading shuffled test images to do prediction"
block = pickle.load(open(expanduser("~/Desktop/Crystal/crystal_data.pkl")))
[_,_,_,_,testData,testLabel] = block

print "compiling network for testing"


model = Sequential()


model.add(Convolution2D(32, 3, 3, input_shape=(3, 224, 224), border_mode='same', activation='relu', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer = l2(0.0)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(64, activation='relu', W_regularizer = l2(0.0)))

model.add(BatchNormalization())
#model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))
# Compile model
epochs = 60
lrate = 0.001
nadam = Nadam()

model.load_weights('model_first.h5')

model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
print(model.summary())

model.summary()

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

te_sample_data, te_sample_label, te_sample_name = sampleSet(testData, testLabel,None,10)

#te_sample_data, te_sample_label  = listOfImagesToNumpy(testData, testLabel)
test_datagen.fit(te_sample_data)

pred = model.predict_generator(test_datagen.flow(te_sample_data, te_sample_label, batch_size=1), val_samples=len(te_sample_data))

correct = 0
confusion = []
total_count = numpy.zeros(4)
confusion_count = numpy.zeros((4,4))
for i in range(len(te_sample_label)):
    total_count[numpy.argmax(te_sample_label[i])] += 1

for i in range(len(te_sample_label)):
	tl=te_sample_label[i]
	p = pred[i]
	if(numpy.argmax(tl) != numpy.argmax(p)):
		print "WRONG " + "true: " + str(numpy.argmax(tl)) + " pred: " + str(numpy.argmax(p)) + " predvec: " + str(p)
	        confusion_count[numpy.argmax(tl),numpy.argmax(p)] += 1	
	else:
		print "CORRECT " + "true: " + str(numpy.argmax(tl)) + " pred: " + str(numpy.argmax(p)) + " predvec: " + str(p) 
	        confusion_count[numpy.argmax(tl),numpy.argmax(p)] += 1	
                correct += 1
print "percent correct: " + str(correct*1.0/len(te_sample_label)) 

print "actual: "
print total_count
print "confusion matrix: "
print confusion_count




