'''
    keep the feature space wide and shallow in the initial stages of the network, 
    and then make it narrower and deeper towards the end.
'''

import os
import sklearn
import librosa

import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt

from PIL              import Image
from jiwer            import wer
from tensorflow       import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

def encode_single_sample(png_file, label):
    ###########################################
    ##  Process the MFCC/Spectogram
    ##########################################
    # 1. Read png file
    image = tf.io.read_file("./mfcc/" + png_file + ".png")
    #image = tf.io.read_file("./spectograms/" + png_file + ".png")
    image = tf.io.decode_png(image, channels=3)
    
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return image, label


# Read metadata file and parse it
dataset = pd.read_csv('dataset.csv', sep="|", index_col=0)
dataset = dataset[["file_name", "normalized_transcription"]]
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Split into train/dev/test sets in 98/1/1
X_train, X_devtest, y_train, y_devtest = train_test_split(
                                                dataset['file_name'], 
                                                dataset['normalized_transcription'], 
                                                test_size=0.02)
X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.5)

print(f"Size of the training set: {len(X_train)}")
print(f"Size of the dev set: {len(X_dev)}")
print(f"Size of the test set: {len(X_test)}")

# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size = {char_to_num.vocabulary_size()})"
)

# Creating tf.Dataset object
batch_size = 32

# Define the trainig dataset
train_dataset = tf.data.Dataset.from_tensor_slices((list(X_train), list(y_train)))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the development dataset
development_dataset = tf.data.Dataset.from_tensor_slices((list(X_dev), list(y_dev)))
development_dataset = (
    development_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((list(X_dev), list(y_dev)))
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
