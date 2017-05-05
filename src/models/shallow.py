
from keras.regularizers import l2
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from utils import *
from os.path import expanduser
from keras.applications import VGG16
import cPickle as pickle
from keras.preprocessing.image import ImageDataGenerator
import numpy
import keras
from keras.models import Model


print "loading shuffled raw train, validation, and test images"
block = pickle.load(open(expanduser("~/Desktop/Crystal/crystal_data.pkl")))
[trainData,trainLabel,validData,validLabel,_,_] = block

img_width, img_height = 224, 224

print "compiling network"
# Create the model
model = Sequential()


model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height), border_mode='same', activation='relu', W_regularizer = l2(0.0)))
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
epochs = 20
lrate = 0.001
nadam = Nadam()
model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
print(model.summary())

# Fit the model
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

tr_sample_data, tr_sample_label, tr_sample_name = sampleSet(trainData,trainLabel,None,10)
va_sample_data, va_sample_label, tr_sample_name = sampleSet(validData,validLabel,None,10)

#tr_sample_data, tr_sample_label = listOfImagesToNumpy(trainData, trainLabel)
#va_sample_data, va_sample_label = listOfImagesToNumpy(validData,validLabel)


train_datagen.fit(tr_sample_data)
valid_datagen.fit(va_sample_data)


#weightings = {0:1.0,1:1.9888268,2:4.0454545,3:3.67010309}
#model.fit_generator(train_datagen.flow(tr_sample_data, tr_sample_label, batch_size=32), 
#        samples_per_epoch=len(tr_sample_data), nb_epoch=epochs, class_weight = weightings,
#        validation_data=valid_datagen.flow(va_sample_data,va_sample_label, batch_size=32),
#        nb_val_samples=len(va_sample_data))

model.fit_generator(train_datagen.flow(tr_sample_data, tr_sample_label, batch_size=32), 
        samples_per_epoch=len(tr_sample_data), nb_epoch=epochs,
        validation_data=valid_datagen.flow(va_sample_data,va_sample_label, batch_size=32),
        nb_val_samples=len(va_sample_data))


model.save_weights("model_first.h5")
