import tensorflow as tf
print(tf.__version__)

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

def get_label(path_name, offset=0):
  start_i = path_name.rfind('/')+1
  end_i = path_name.find('_')
  label = int(path_name[start_i:end_i])
  label -= offset
  return label

def load_glyph_img(path_name):

  img = tf.io.read_file(path_name)
  img = tf.image.decode_png(img, channels=3)
  img = tf.image.rgb_to_grayscale(img)
  img = tf.squeeze(img)
  img = tf.cast(img, tf.float32)
  img /= 255.0  # normalize pixels to 0,1

  return img

def load_glyphs(path):

  imgs = []
  labels = []
  for path_name in  os.listdir(path):
    if 'png' in path_name:
      img = load_glyph_img(path+path_name)
      label = get_label(path_name, offset=ord('א'))

      imgs.append(img)
      labels.append(label)

  imgs = np.array(imgs)
  labels = np.array(labels)

  return labels, imgs

if __name__=="__main__":
    pass