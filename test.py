#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import numpy as np

import sys
from tqdm import tqdm
import hashlib
import os
import glob
import random

import argparse

from models import create_se_convlstm_model, create_vanilla, create_convlstm_model

def is_training(file_name):
    file_name = os.path.basename(file_name)
    hash_object = hashlib.md5(file_name.encode())
    md5_hash = hash_object.hexdigest()
    simple_hash = int(md5_hash[-4:], 16)%1000
    return simple_hash>250


def predict_file(file_path, model, n_frames=3):
    video = np.load(file_path)
    #print(video.shape)
    video = video[:n_frames*(len(video)//n_frames)]
    #print(video.shape)

    inputs = []

    for i in range(n_frames):
        inputs.append(video[i::n_frames])

    #print(file_path, len(inputs), inputs[0].shape)
    results = model.predict(inputs)
    results = results.sum(axis=0)
    return results.argmax()


def get_acc(file_path, model, n_frames=3):
    pred = predict_file(file_path, model, n_frames)

    base = os.path.basename(file_path)
    class_ = int(base.split('.')[0].split('_')[-1]) - 1

    return pred == class_


def caluclate_accuracy(files_list, model, n_frames=3):
    preds = []
    for  file_path in tqdm(files_list):
        preds.append(get_acc(file_path, model,n_frames))

    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('--n_frames', type=int, default=3)
    parser.add_argument('--model')
    parser.add_argument('--model_path')

    args = parser.parse_args()
    FRAMES_DATA_FOLDER = args.data_folder

    files_list = [file for file in glob.glob(os.path.join(FRAMES_DATA_FOLDER, '*', '*.npy')) if is_training(file) == False]

    if args.model == 'se':
        model = create_se_convlstm_model(args.n_frames)
    elif args.model == 'vanilla':
        model = create_vanilla(args.n_frames)
    elif args.model == 'vanilla':
        model = create_convlstm_model(args.n_frames)

    model.load_weights(args.model_path)

    print('loaded the model')

    accs = caluclate_accuracy(files_list, model, n_frames=3)

    print('model accuracy is {}'.format(sum(accs)/len(accs)))
