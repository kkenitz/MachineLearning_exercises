# -*- coding: utf-8 -*-
"""
Created on Tue Nov 09 11:23:54 2021

@author: Kash
"""

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
  # Read the file:
  # - ignore the headers
  # - read in the array pixel values 
  # - reshape the pixel array to image shape
    with open(filename) as training_file:
        images=[]
        labels=[]
        reader = csv.reader(training_file, delimiter=",")
        next(reader,None)
       
        for row in reader:
            label = row[0]
            labels.append(label)
            tmp=np.array(row[1:]).astype(float).reshape((28,28))
            images.append(tmp)
            
    labels = np.array(labels).astype(float)
    images = np.array(images).astype(float)

    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    labels = labels[randomize]
    images = images[randomize]
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

# Create an ImageDataGenerator and do Image Augmentation
#train_datagen = ImageDataGenerator(
#    rescale = 1./255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode='nearest'
#)

# better performance without augmentation
train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(training_labels))+1, activation='softmax')
])

# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the Model
ind=round(0.8*len(training_labels))
training_generator = train_datagen.flow(training_images[:ind], training_labels[:ind], batch_size=32)
validation_generator = validation_datagen.flow(training_images[ind:], training_labels[ind:], batch_size=32)

history = model.fit_generator(training_generator, steps_per_epoch=len(training_images)/32, epochs=2, validation_data = validation_generator, verbose = 1)

model.evaluate(testing_images, testing_labels, verbose=0)

