# -*- coding: utf-8 -*-
"""
Created on Tue Nov 09 11:15:22 2021

@author: Kash
"""
import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DESIRED_ACCURACY = 0.999

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=DESIRED_ACCURACY):
        print("\nReached 99.9% accuracy so cancelling training!")
        self.model.stop_training = True
       

callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])


model.compile(loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy'])
    
train_datagen = ImageDataGenerator(rescale=1/255)

# Please use a target_size of 150 X 150.
train_generator = train_datagen.flow_from_directory(
    '/tmp/h-or-s',  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 300x300
    batch_size=20,
    class_mode='binary')
    # Expected output: 'Found 80 images belonging to 2 classes'

history = model.fit_generator(
    train_generator,
    steps_per_epoch=4,
    epochs=20,
    verbose=1,
    callbacks=[callbacks])