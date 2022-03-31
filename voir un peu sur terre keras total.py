#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:59:52 2021

@author: harry
"""
# Les modules indispensables pour trouver le modèle optimal.

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from joblib import dump
from keras.models import load_model
import gzip
import glob

import Meteo_terre_traitement_explicatif as mttt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.utils import class_weight
import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# On importe les données csv de MeteoNet(les trois fichiers csv de 2016, 2017, 2018).
"""

chunksize=100000
chunks=[]
'''
for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2016.csv',chunksize=chunksize):
    chunks.append(chunk)

for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2017.csv',chunksize=chunksize):
    chunks.append(chunk)
'''
for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2018.csv',chunksize=chunksize):
    chunks.append(chunk)


data=pd.concat(chunks, axis=0)



    
# Une fois les données importées on peut crée un objet appelé 'meteo' à partir du module 'meteo_terre_traitement_total'.

meteo=mttt.Meteo_Data_View(data, jours=3, var_corbeille=['point_rosée', 'pluie'], cible='température', vue=24)



dataset=meteo.data_good_shape
cible=meteo.target

cible=pd.DataFrame(cible)
print(cible.value_counts(normalize=True))
cible=np.array(cible)

np.save("dataset_3_jours_point_rosée_pluie_viré_température_2018_24h", dataset)
np.save("labels_3_jours_point_rosée_pluie_viré_température_2018_24h", cible)

"""


dataset_2016=np.load("dataset_3_jours_point_rosée_pluie_viré_humidité_2016_24h.npy")
cible_2016=np.load("labels_3_jours_point_rosée_pluie_viré_humidité_2016_24h.npy")

dataset_2017=np.load("dataset_3_jours_point_rosée_pluie_viré_humidité_2017_24h.npy")
cible_2017=np.load("labels_3_jours_point_rosée_pluie_viré_humidité_2017_24h.npy")

dataset_2018=np.load("dataset_3_jours_point_rosée_pluie_viré_humidité_2018_24h.npy")
cible_2018=np.load("labels_3_jours_point_rosée_pluie_viré_humidité_2018_24h.npy")




dataset=np.concatenate([dataset_2016, dataset_2017, dataset_2018], axis=0)
cible=np.concatenate([cible_2016, cible_2017, cible_2018], axis=0)



print(np.unique(cible))
'''
encoder=LabelEncoder()
cible=encoder.fit_transform(cible)
'''


X_train, X_test, y_train, y_test=train_test_split(dataset, cible, test_size=0.2, random_state=38)

X_val, X_eval, y_val, y_eval=train_test_split(X_test, y_test, test_size=0.5, random_state=38)
treat=RobustScaler()
treat.fit(X_train)
X_train_treat=treat.transform(X_train)
X_val_treat=treat.transform(X_val)
X_eval_treat=treat.transform(X_eval)

print(np.unique(y_train))

np.save('data_test_sequential_humidite.npy', X_eval)
'''
dump(encoder, 'encodeur_température_24h.joblib')
'''
dump(treat, 'preprocessor_humidite_24h.joblib')
np.save('data_test_cible_humidite.npy', y_eval)

'''
def build_model(n_hidden=4, n_neurons=600, learning_rate=0.001, input_shape=X_train_treat.shape[1], dropout=0.2):

  model=keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=(input_shape,)))
  for layer in range(n_hidden):
      model.add(keras.layers.Dense(n_neurons, activation='relu'))
  model.add(keras.layers.Dropout(dropout))
  model.add(keras.layers.Dense(8, activation='softmax'))
  optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

keras_cla=keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, epochs=100)

stop=EarlyStopping(monitor='val_loss', mode='min', patience=20)

param_grid={'dropout':[0.1, 0.5]}



search=GridSearchCV(keras_cla, param_grid, cv=3)
search.fit(X_train_treat, y_train, epochs=100, validation_data=(X_val_treat, y_val), callbacks=stop)
print(search.best_params_)
print(search.best_score_)
'''
model=keras.models.Sequential()

model.add(keras.layers.InputLayer(input_shape=(X_train_treat.shape[1], )))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', metrics=['mae'], optimizer=keras.optimizers.Adam(learning_rate=0.0001))
stop=keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4)
'''
class_weights=class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
weights={i:j for i, j in zip(np.unique(y_train, class_weights))}
'''
history=model.fit(X_train_treat, y_train, epochs=80, validation_data=(X_val_treat, y_val), callbacks=stop)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 2)
plt.show()

model.save('layers_humidite_24h.h5')
print(model.evaluate(X_eval_treat, y_eval))


"""
param_grid={'n_estimators':[300], 'max_features':[40]}



search=GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
print(search.score(X_test, y_test))


search=RandomForestClassifier()
search.fit(X_train, y_train)
print(search.score(X_test, y_test))

def plot_var_importantes(model):
    n_features=dataset.shape[1]
    plt.figure(figsize=[12, 8])
    plt.barh(range(n_features), search.feature_importances_, align='center')
    plt.yticks(np.arange(n_features))
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    
plot_var_importantes(search)


''' calcule la certitude de la prévision'''

probas=search.predict_proba(X_test)
max_liste=[]
for line in probas:
    maximum=np.max(line)
    max_liste.append(maximum)
certitude=np.mean(max_liste)

print(certitude)

"""
