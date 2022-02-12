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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            

# Read metadata file and shuffle it
dataset = pd.read_csv('dataset.csv', sep="|", index_col=0)
dataset = dataset[["file_name", "normalized_transcription"]]
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Parsing metadata
dataset['file_name'] = dataset['file_name'] + ".png"
dataset['normalized_transcription'] = dataset['normalized_transcription'].str.lower()
dataset['normalized_transcription'] = dataset['normalized_transcription'].apply(lambda row : tf.strings.unicode_split(row, input_encoding="UTF-8"))
dataset['normalized_transcription'] = dataset['normalized_transcription'].apply(lambda row : char_to_num(row))
dataset['normalized_transcription'] = dataset['normalized_transcription'].apply(np.array)
dataset['normalized_transcription'] = dataset['normalized_transcription'].apply(list)

# Split into train/dev/test sets in 98/1/1
df_train, df_devtest = train_test_split(dataset, test_size=0.02)
df_dev, df_test = train_test_split(df_devtest, test_size=0.5)

print(f"\n\nSize of the training set: {len(df_train)}")
print(f"Size of the dev set: {len(df_dev)}")
print(f"Size of the test set: {len(df_test)}\n\n")

print(
    f"\nThe vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size = {char_to_num.vocabulary_size()})"
)

# Creating tf.Dataset object
batch_size = 128
datagen = ImageDataGenerator(preprocessing_function=encode_single_sample)

# Define the trainig dataset
train_dataset = datagen.flow_from_dataframe(
    dataframe=df_train, 
    directory=".\mfcc", 
    x_col="file_name", 
    y_col="normalized_transcription", 
    class_mode="categorical",
    target_size=(496, 369), 
    batch_size=batch_size)

# Define the development dataset
development_dataset = datagen.flow_from_dataframe(
    dataframe=df_dev, 
    directory=".\mfcc", 
    x_col="file_name", 
    y_col="normalized_transcription", 
    class_mode="categorical",
    target_size=(496, 369), 
    batch_size=batch_size)

# Define the test dataset
test_dataset = datagen.flow_from_dataframe(
    dataframe=df_test, 
    directory=".\mfcc", 
    x_col="file_name", 
    y_col="normalized_transcription", 
    class_mode="categorical",
    target_size=(496, 369), 
    batch_size=batch_size)

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

# Define the number of epochs
EPOCHS = 1
STEP_SIZE_TRAIN = train_dataset.n // train_dataset.batch_size
STEP_SIZE_VALID = development_dataset.n // development_dataset.batch_size

# Callback function to check transcription on the val set.
validation_callback = CallbackEval(development_dataset)
# Train the model
model.fit_generator(
    generator=train_dataset,
    validation_data=development_dataset,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_steps=STEP_SIZE_VALID,
    callbacks=[validation_callback], 
    epochs=EPOCHS
    )