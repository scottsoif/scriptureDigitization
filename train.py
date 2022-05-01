import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import datetime


import numpy as np
import pathlib
import os

from sklearn.model_selection import train_test_split

from preprocess import load_glyphs

def gen_data(file_dir):
    labels, imgs = load_glyphs(file_dir)


    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels)  
    # train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

    # train_imgs_ds = tf.data.Dataset.from_tensor_slices(train_imgs)
    # train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)
    # train_ds = tf.data.Dataset.zip((train_imgs_ds, train_labels_ds))
    # train_ds = train_ds.batch(16)

    train_imgs = tf.expand_dims(train_imgs, 3)
    test_imgs = tf.expand_dims(test_imgs, 3)

    train_datagen = ImageDataGenerator(
        # rotation_range=3,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.5,
        # zoom_range=0.2,
        # fill_mode='nearest'
        )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_imgs, train_labels, shuffle=True, batch_size=16)
    test_generator = train_datagen.flow(test_imgs, test_labels, batch_size=16)

    return train_generator, test_generator


def build_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dense(27, activation='softmax')
        ]
    )

    return model

def plot(history, save=True):
  
  # The history object contains results on the training and test
  # sets for each epoch
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get the number of epochs
  epochs = range(len(acc))

  fig1 = plt.figure()
  plt.title('Training and validation accuracy')
  plt.plot(epochs, acc, color='blue', label='Train')
  plt.plot(epochs, val_acc, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  fig2 = plt.figure()
  plt.title('Training and validation loss')
  plt.plot(epochs, loss, color='blue', label='Train')
  plt.plot(epochs, val_loss, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  if save:
    timestamp = datetime.now().strftime('%Y%m%d_%I.%M.%S %p')
    fig1.savefig(f'outputs/acc_graph_{timestamp}.png')
    fig2.savefig(f'outputs/loss_graph_{timestamp}.png')



def train(data_file_dir):

    train_generator, test_generator = gen_data(data_file_dir)

    model = build_model()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # history = model.fit(train_ds, validation_data=(test_imgs, test_labels), epochs=10)
    history = model.fit(train_generator, validation_data=test_generator, epochs=15,
            steps_per_epoch=train_generator.n//train_generator.batch_size,
            validation_steps=test_generator.n//test_generator.batch_size)

    # model.save("/content/ocr_model_english.h5")
    model.save("outputs/ocr_model_hebrew.h5")

    return history


def main():

    # history = train(data_file_dir="data/heb_dataset/")
    # plot(history, save=True)
    pass


if __name__=="__main__":
    main()
