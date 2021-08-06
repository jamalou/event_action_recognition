#!/usr/bin/env python
# coding: utf-8



import aedat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

import aer
import cv2
import numpy as np

import matplotlib.pyplot as plt
import sys

import hashlib
import os
import glob
import random

import argparse


def is_training(file_name):
    file_name = os.path.basename(file_name)
    hash_object = hashlib.md5(file_name.encode())
    md5_hash = hash_object.hexdigest()
    simple_hash = int(md5_hash[-4:], 16)%1000
    return simple_hash>250




def se_block(inputs, ch, ratio=16):
    gap_layer = layers.GlobalAveragePooling2D()
    ds_1_layer = layers.Dense(ch//ratio, activation='relu')
    ds_2_layer = layers.Dense(ch, activation='sigmoid')
    mp_layer = layers.Multiply()
    
    
    X = [gap_layer(inp) for inp in inputs]
    X = [ds_1_layer(inp) for inp in X]
    X = [ds_2_layer(inp) for inp in X]
    X = [mp_layer([inp, x]) for inp, x in zip(inputs, X)]
    
    return X


def cnn_multiple_images(inputs, n_filters, se_ratio=16):
    
    conv_1 = layers.Conv2D(n_filters, 3, activation='relu', padding='same')
    conv_2 = layers.Conv2D(n_filters, 3, activation='relu', padding='same')
    bn_ = layers.BatchNormalization()
    
    X = [conv_1(inp) for inp in inputs]
    X = [conv_2(inp) for inp in X]
    X = [bn_(inp) for inp in X]
    X = se_block(X, ch=n_filters, ratio=se_ratio)
    
    
    return X
	
	
def create_model(n_frames=3):
	inputs = [layers.Input(shape=(260, 346, 3)) for _ in range(n_frames)]
	X = cnn_multiple_images(inputs, n_filters=16, se_ratio=16)

	#x_1, x_2, x_3 = cnn_multiple_images(x_1, x_2, x_3, n_filters=32, se_ratio=16)
	#x = layers.Concatenate()([x_1, x_2, x_3])

	X = tf.keras.backend.stack(
		X,
		axis=-1
	)

	X = layers.ConvLSTM2D(
		filters=64,
		kernel_size=(5, 5),
		padding="same",
		return_sequences=True,
		activation="relu",
	)(X)

	X = layers.BatchNormalization()(X)

	X = layers.Conv3D(
		filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
	)(X)

	X = layers.GlobalAveragePooling3D()(X)
	X = layers.Dense(units=512, activation="relu")(X)
	X = layers.Dropout(0.3)(X)

	outputs = layers.Dense(units=10, activation="softmax")(X)
	# Next, we will build the complete model and compile it."""

	model = keras.models.Model(inputs, outputs)
	return model


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, data_folder, to_fit=True, batch_size=32, n_classes=10, shuffle=True, 
                 dim=(260, 346), n_channels=3, n_frames=20, is_train=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.files_list = [file for file in glob.glob(data_folder) if is_training(file) == is_train]*10
        self.to_fit = to_fit
        self.dim = dim
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(10*len(self.files_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X ,y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.files_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_frames(self, file_name):
        
        x = np.load(file_name)
        rdm_idx = random.randint(0, len(x)-self.n_frames-1)
        chunck = x[rdm_idx: rdm_idx + self.n_frames]
        
        return chunck #/max_
        
        
    def _generate_X(self, list_IDs_temp):
        'Generates data containing batch_size images'
        # Initialization
        if self.n_frames == 1:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        if self.n_frames != 1:
            X = []
            for i in range(self.n_frames):
                X.append(np.empty((self.batch_size, *self.dim, self.n_channels)))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            frames = self._generate_frames(ID)
            for frame_no in range(self.n_frames):
                X[frame_no][i,] = frames[frame_no,]

        return X
    
    def _get_class(self, file_name):
        
        base = os.path.basename(file_name)
        class_ = int(base.split('.')[0].split('_')[-1]) - 1
        label = np.zeros((1, self.n_classes))
        label[0, class_] = 1
        return label

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._get_class(ID)

        return y


if __name__ == '__main__':
    
	parser = argparse.ArgumentParser()
	parser.add_argument('data_folder')
	parser.add_argument('--n_frames', type=int, default=3)
	args = parser.parse_args()
	FRAMES_DATA_FOLDER = args.data_folder

	print(FRAMES_DATA_FOLDER)

	model = create_model(args.n_frames)
	model.compile(
		loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
	)

	print('created the model')


	training_generator   = DataGenerator(os.path.join(FRAMES_DATA_FOLDER, '*/*.npy'), is_train=True, batch_size=2, n_frames=args.n_frames)
	validation_generator = DataGenerator(os.path.join(FRAMES_DATA_FOLDER, '*/*.npy'), is_train=False, batch_size=2, n_frames=args.n_frames)
	# Define some callbacks to improve training.
	early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
	model_checkpoint = keras.callbacks.ModelCheckpoint('Model_se_conv_lstm.h5', save_best_only=True)

	# Define modifiable training hyperparameters.
	epochs = 5

	hist = model.fit(training_generator, validation_data=validation_generator, workers=4,
					 epochs=epochs, callbacks=[early_stopping, reduce_lr, model_checkpoint], use_multiprocessing=True)


	model.save('Model_final.h5')

