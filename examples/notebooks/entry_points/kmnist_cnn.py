#!/usr/bin/env python

import argparse
from datetime import date
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                            MaxPool2D)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam as optimizer
from tensorflow.python.keras.utils import to_categorical

PWD = Path.cwd()
DATA_DIR = PWD.joinpath('..', 'data').resolve()
KMNIST_DATA_DIR = DATA_DIR.joinpath('kmnist')

TEST_DATA_OFFSET = 5000

EPOCHS = 100
BATCH_SIZE = 50
NUM_CLASSES = 10


def get_args() -> argparse.Namespace:
    """Get arguments from command line."""
    parser = argparse.ArgumentParser()
    # Hyperparams specified when executing estimator
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('-n', '--num-classes', type=int, default=NUM_CLASSES)
    # Dirs for model, train, validation and output data
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test-dir', type=str,
                        default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--output-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    # Args read from command line
    args, _ = parser.parse_known_args()
    return args


def _normalize_images(X: np.ndarray) -> np.ndarray:
    """Normalize the pixel values (0 to 255) in between 0 to 1."""
    return X / 255


def _reshape_images(X: np.ndarray) -> np.ndarray:
    """
    Reshape and convert the collection of matrices to the one of vectors.
    """
    return X.reshape(X.shape + (1,))


def preprocess_images(X: np.ndarray, reshape: bool = False) -> np.ndarray:
    """Normalize and reshape images."""
    X = _normalize_images(X)
    if reshape:
        X = _reshape_images(X)
    return X


def get_input_shape(X: np.ndarray, channel: int = 1) -> tuple:
    """Get input shape for the model to be trained."""
    _, height, width, channel = X.shape
    return height, width, channel


def get_input_tensor(X: np.ndarray, channel: int = 1) -> Input:
    """Get input tensor for the model to be trained."""
    _, height, width, channel = X.shape
    return Input(shape=(height, width, channel))


def get_model(input_shape: tuple, num_classes: int) -> Sequential:
    """Define the convolutional nueral network model."""
    model = Sequential()
    model.add(Conv2D(6, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    # Set params and dirs for training
    args = get_args()
    x_train_path = os.path.join(args.train_dir, 'kmnist-train-imgs.npz')
    y_train_path = os.path.join(args.train_dir, 'kmnist-train-labels.npz')
    x_test_path = os.path.join(args.test_dir, 'kmnist-test-imgs.npz')
    y_test_path = os.path.join(args.test_dir, 'kmnist-test-labels.npz')

    # Load train data
    X_train = np.load(x_train_path)['arr_0']
    y_train = np.load(y_train_path)['arr_0']

    # Load test data
    X_test = np.load(x_test_path)['arr_0']
    y_test = np.load(y_test_path)['arr_0']

    # Normalize (and reshape) images
    X_train = preprocess_images(X_train, reshape=True)
    X_test = preprocess_images(X_test, reshape=True)

    # One-hot encoding
    num_classes = args.num_classes
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Configure and compile model
    input_tensor = get_input_tensor(X_train)
    input_shape = get_input_shape(X_train)
    model = get_model(input_shape, num_classes)
    model.compile(optimizer(lr=1e-5), loss='categorical_crossentropy',
                  metrics=['acc'])

    # Partition validation data from test data
    validation_data = X_test[:TEST_DATA_OFFSET], y_test[:TEST_DATA_OFFSET]

    # Train the model with train/validation data
    batch_size = args.batch_size
    epochs = args.epochs
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=validation_data)

    # Save the trained model
    today = date.today().isoformat()
    dirpath = os.path.join(args.model_dir, today)
    tf.contrib.saved_model.save_keras_model(model, dirpath)
