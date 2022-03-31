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
from joblib import dump, load

import gzip
import glob

import Meteo_terre_traitement_explicatif as mttt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE



# On importe les données csv de MeteoNet(les trois fichiers csv de 2016, 2017, 2018).

"""
chunksize=100000
chunks=[]

for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2016.csv',chunksize=chunksize):
    chunks.append(chunk)

for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2017.csv',chunksize=chunksize):
    chunks.append(chunk)

for chunk in pd.read_csv('/home/harry/Documents/projet_station_meteo/NW_Ground_Stations/Meteo_NW_2018.csv',chunksize=chunksize):
    chunks.append(chunk)

data=pd.concat(chunks, axis=0)



    
# Une fois les données importées on peut crée un objet appelé 'meteo' à partir du module 'meteo_terre_traitement_total'.

meteo=mttt.Meteo_Data_View(data, jours=3, var_corbeille=['point_rosée', 'pluie'], cible='force', vue=24)



dataset=meteo.data_good_shape
cible=meteo.target

cible=pd.DataFrame(cible)
print(cible.value_counts(normalize=True))

cible=np.array(cible)
print(np.unique(cible))



np.save("dataset_3_jours_point_rosée_pluie_viré_force_2018_24h", dataset)
np.save("labels_3_jours_point_rosée_pluie_viré_force_2018_24h", cible)

"""

dataset_2016=np.load("dataset_3_jours_point_rosée_pluie_viré_direction_2016_24h.npy")
cible_2016=np.load("labels_3_jours_point_rosée_pluie_viré_direction_2016_24h.npy")

dataset_2017=np.load("dataset_3_jours_point_rosée_pluie_viré_direction_2017_24h.npy")
cible_2017=np.load("labels_3_jours_point_rosée_pluie_viré_direction_2017_24h.npy")

dataset_2018=np.load("dataset_3_jours_point_rosée_pluie_viré_direction_2018_24h.npy")
cible_2018=np.load("labels_3_jours_point_rosée_pluie_viré_direction_2018_24h.npy")
'''
weak=np.load('weak_var.npy')
'''

dataset=np.concatenate([dataset_2016, dataset_2017, dataset_2018], axis=0)
cible=np.concatenate([cible_2016, cible_2017, cible_2018], axis=0)
'''
dataset=pd.DataFrame(dataset)
dataset=dataset.drop(weak, axis=1)
dataset=np.array(dataset)
'''

''' pour éviter le déséquilibre des données pour la pluie notamment '''
"""
dataset=pd.DataFrame(dataset)
cible=pd.DataFrame(cible)

data_table=pd.concat([dataset, cible], axis=1)

data_table_1=data_table[data_table.iloc[:,-1]==1]
data_table_0=data_table[data_table.iloc[:,-1]==0]

data_table_0=data_table_0.iloc[:data_table_1.shape[0], :]

data=pd.concat([data_table_0, data_table_1], axis=0)

dataset=np.array(data.iloc[:,:-1])
cible=np.array(data.iloc[:,-1])

"""
  



X_train, X_test, y_train, y_test=train_test_split(dataset, cible, test_size=0.2, random_state=17, stratify=cible)

X_val, X_eval, y_val, y_eval=train_test_split(X_test, y_test, test_size=0.5, random_state=17, stratify=y_test)

np.save('data_test_direction.npy', X_eval)

model=RandomForestClassifier(n_estimators=40, max_depth=20, max_leaf_nodes=100000, criterion='entropy', class_weight='balanced')

model.fit(X_train, y_train)

print(model.score(X_val, y_val))


print(classification_report(y_eval, model.predict(X_eval)))


dump(model, 'direction_24_heures_random_forest.joblib')

def plot_var_importantes(model):
    n_features=dataset.shape[1]
    plt.figure(figsize=[30, 30])
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features))
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    
plot_var_importantes(model)


        

''' calcule la certitude de la prévision'''

probas=model.predict_proba(X_test)
max_liste=[]
for line in probas:
    maximum=np.max(line)
    max_liste.append(maximum)
certitude=np.mean(max_liste)

print(certitude)

weak_features=[]

for feat in model.feature_importances_:
    
    if feat < 0.0023:
        
        weak_features.append(list(model.feature_importances_).index(feat))
   
np.save('weak_var.npy', weak_features)

