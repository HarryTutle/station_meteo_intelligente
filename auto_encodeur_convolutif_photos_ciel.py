#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:08:45 2023

@author: harry
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from tensorflow.keras import optimizers, layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.keras import callbacks
from keras.models import load_model

from PIL import Image, ImageEnhance
import glob
import os
import shutil
from tensorflow.keras.preprocessing.image import load_img

from sklearn.decomposition import PCA






encodeur_conv=models.Sequential([
    layers.Reshape([64, 64, 1], input_shape=[64, 64]),
    layers.Conv2D(40, kernel_size=3, padding="same", activation='selu'),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(80, kernel_size=3, padding="same", activation="selu"),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(120, kernel_size=3, padding="same", activation='selu'),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(240, kernel_size=3, padding="same", activation='selu'),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(480, kernel_size=3, padding='same', activation='selu'),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(960, kernel_size=3, padding="same", activation="selu"),
    layers.MaxPool2D(pool_size=2),
    ])
    
decodeur_conv=models.Sequential([
    layers.Conv2DTranspose(480, kernel_size=3, strides=2, activation="selu", padding="same", input_shape=[1, 1, 960]),
    layers.Conv2DTranspose(240, kernel_size=3, strides=2, padding="same", activation="selu"),
    layers.Conv2DTranspose(120, kernel_size=3, strides=2, padding="same", activation="selu"),
    layers.Conv2DTranspose(80, kernel_size=3, strides=2, padding="same", activation="selu"),
    layers.Conv2DTranspose(40, kernel_size=3, strides=2, padding="same", activation="selu"),
    layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
    layers.Reshape([64, 64])
    ])

autoencodeur_conv=models.Sequential([encodeur_conv, decodeur_conv])

encodeur_summary=encodeur_conv.summary()
decodeur_summary=decodeur_conv.summary()

X_train_debug=np.load("/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_one/tenseur.npy")
index_train_debug=np.load("/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_one/index_tenseur.npy")


X_test_debug=np.load("/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_two/tenseur.npy")
index_test_debug=np.load("/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_two/index_tenseur.npy")


callback=callbacks.EarlyStopping(monitor='loss', patience=3, mode='min')

autoencodeur_conv.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001), metrics='accuracy')

history=autoencodeur_conv.fit(X_train_debug, X_train_debug, epochs=20, validation_data=(X_test_debug, X_test_debug), batch_size=32, callbacks=[callback])

print(X_test_debug.shape)
print(X_test_debug[10].shape)

debug_result=encodeur_conv.predict(X_test_debug)[:40]

debug_result_2=decodeur_conv.predict(debug_result)*255.0

decodeur_conv.save('/home/harry/station meteo/analyse photos ciel/decodeur_convolutif.h5')
encodeur_conv.save('/home/harry/station meteo/analyse photos ciel/encodeur_convolutif.h5')
autoencodeur_conv.save('/home/harry/station meteo/analyse photos ciel/autoencodeur_convolutif.h5')


plt.figure(figsize=(20, 20))
for i in range(0, 20):
    img=debug_result_2[i]
    img=Image.fromarray(img)
    plt.subplot(10, 2, i+1)
    plt.xlabel(index_test_debug[i])
    plt.subplots_adjust(hspace=(0.6))
    plt.imshow(img)


