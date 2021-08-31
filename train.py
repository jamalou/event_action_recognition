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

import models #import create_se_convlstm_model, create_vanilla, create_convlstm_model, create_xception_convlstm_model
from data_generators import DataGeneratorMultipleInput, DataGeneratorSingleInput

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('--n_frames', type=int, default=3)
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    FRAMES_DATA_FOLDER = args.data_folder

    print(FRAMES_DATA_FOLDER)

    if args.model == 'se':
        model = models.create_se_convlstm_model(args.n_frames)
        training_generator   = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=True, batch_size=args.batch_size, n_frames=args.n_frames)
        validation_generator = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=False, batch_size=args.batch_size, n_frames=args.n_frames)
    elif args.model == 'xlstm':
        model = models.create_xception_convlstm_model(args.n_frames)
        training_generator   = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=True, batch_size=args.batch_size, n_frames=args.n_frames)
        validation_generator = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=False, batch_size=args.batch_size, n_frames=args.n_frames)

    elif args.model == 'vanilla':
        model = models.create_vanilla(args.n_frames)
        training_generator   = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=True, batch_size=args.batch_size, n_frames=args.n_frames)
        validation_generator = DataGeneratorMultipleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=False, batch_size=args.batch_size, n_frames=args.n_frames)

    elif args.model == 'clstm':
        model = models.create_convlstm_model(args.n_frames)
        training_generator   = DataGeneratorSingleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=True, batch_size=args.batch_size, n_frames=args.n_frames)
        validation_generator = DataGeneratorSingleInput(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy'), is_train=False, batch_size=args.batch_size, n_frames=args.n_frames)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(learning_rate=args.lr),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    print('created the model')


    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
    model_checkpoint = keras.callbacks.ModelCheckpoint(f'Model_{args.model}.h5', save_best_only=True)

    # Define modifiable training hyperparameters.
    epochs = args.epochs

    hist = model.fit(training_generator, validation_data=validation_generator, workers=4,
                     epochs=epochs, callbacks=[early_stopping, reduce_lr, model_checkpoint], use_multiprocessing=False)


    model.save('Model_final.h5')
