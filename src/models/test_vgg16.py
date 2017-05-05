
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
sampling = False

img_width, img_height = 224, 224

print "compiling network"

model = VGG16(weights='imagenet',include_top=True)
vgg_body = model.layers[-1].output
softmax_layer = keras.layers.core.Dense(4,W_regularizer=l2(0.01),init='glorot_uniform',activation='softmax')(vgg_body)
tl_model = Model(input=model.input, output=softmax_layer)
tl_model.load_weights("model_after_first_run_crystal_four.h5")

for layer in model.layers:
    layer.trainable = False

tl_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Nadam(),
              metrics=['accuracy'])

tl_model.summary()

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

#te_sample_data, te_sample_label, te_sample_name = sampleSet(testData, testLabel,None, 1)

if(sampling == False):
	te_sample_data, te_sample_label = listOfImagesToNumpy(testData, testLabel)
else:
	te_sample_data, te_sample_label, te_sample_name = sampleSet(testData, testLabel,None, 1)
te_sample_data = numpy.transpose(te_sample_data,[0,2,3,1])

print "METRICS 0 "
print numpy.shape(te_sample_data)
print numpy.shape(te_sample_label)

test_datagen.fit(te_sample_data)

pred = tl_model.predict_generator(test_datagen.flow(te_sample_data, te_sample_label), val_samples=len(te_sample_data))

correct = 0
confusion = []
total_count = numpy.zeros(4)
confusion_count = numpy.zeros((4,4))
for i in range(len(te_sample_label)):
    total_count[numpy.argmax(te_sample_label[i])] += 1


for i in range(len(te_sample_label)):
	tl=te_sample_label[i]
	p = pred[i]
	print "METRICS"
	print numpy.shape(pred)
	print numpy.shape(pred[i])
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
