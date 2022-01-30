'''
    keep the feature space wide and shallow in the initial stages of the network, 
    and then make it narrower and deeper towards the end.
'''

import warnings
import numpy             as np
import pandas            as pd
import tensorflow        as tf
import matplotlib.pyplot as plt

from NN               import *
from jiwer            import wer
from tensorflow       import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


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

print(f"\n\nSize of the training set: {len(X_train)}")
print(f"Size of the dev set: {len(X_dev)}")
print(f"Size of the test set: {len(X_test)}\n\n")

print(
    f"\nThe vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size = {char_to_num.vocabulary_size()})"
)

# Creating tf.Dataset object
batch_size = 12

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

print(train_dataset)
print(development_dataset)
print(test_dataset)

# Create model
model = NN.build_model(
    input_dim = (496, 369, 3),
    output_dim = char_to_num.vocabulary_size(),
    rnn_units = 512,
)
model.summary(line_length=110)

# Define the number of epochs.
epochs = 1
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(development_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=development_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)