#!/usr/bin/env

"""
- Provides building,training, and testing of the final model described in:

Forecasting Outcomes in Rheumatoid Arthritis Using Longitudinal Deep Learning on Electronic Health Record Data (Norgeot et al, 2018)


- Associated patient data is not provided for privacy reasons
- Input data is expected in typical 3d timeseries form: samples, timesteps, features

"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_curve, auc

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional ,GRU,Conv1D, Flatten, MaxPooling1D
from keras import layers
from keras.layers import GRU, Dense, Bidirectional, Embedding, Dropout, TimeDistributed

from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt
import sys

import numpy


# load data:
train_data = numpy.load('ucsf_train_data.npy')
train_labels = numpy.load('ucsf_train_labels.npy')

val_data = numpy.load('ucsf_val_data.npy')
val_labels = numpy.load('ucsf_val_labels.npy')

test_data = numpy.load('ucsf_test_data.npy')
test_labels = numpy.load('ucsf_test_labels.npy')

# reshuffle val data/labels (while preserving mapping between them) can be useful if using cv
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


val_data, val_labels = unison_shuffled_copies(val_data, val_labels)

print("train_data shape: ",train_data.shape)

#Define variables
timesteps = train_data.shape[1]
features = train_data.shape[2]

#Define Model
model = Sequential()
model.add(TimeDistributed(Dense(3),input_shape=(timesteps,features)))
model.add(Activation('relu'))

model.add(GRU(32,recurrent_dropout = 0.3,
              kernel_regularizer=l2(0.00674),
              recurrent_regularizer=l2(0.00544)))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(layers.Dense(32,kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(layers.Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy','binary_crossentropy'])

# Fit model
history = model.fit(train_data, train_labels,
          batch_size=64,
          epochs=304,
          shuffle = True,
          validation_data = (val_data,val_labels),          
          verbose = 1)

# plot training performance
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
loss_history.append(loss)
val_loss_history.append(val_loss)

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo')
plt.plot(epochs, val_acc, 'g')
plt.title('Training and validation accuracy')
plt.legend(['train', 'val'], loc='lower right')
plt.figure()

plt.plot(epochs, loss, 'bo')
plt.plot(epochs, val_loss, 'g')
plt.title('Training and validation loss')
plt.legend(['train', 'val'], loc='upper right')
plt.show(block=True)

# Note:
# For optimal model performance, you should retain the model on the combined train/val here

# Use trained model to predict on test data
probs = model.predict_proba(test_data)
fpr, tpr, threshold = roc_curve(test_labels,probs)
auc_score = auc(fpr, tpr)

print('test auroc: ',auc_score)

