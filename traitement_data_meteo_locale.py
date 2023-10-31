#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:38:58 2023

@author: harry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from tensorflow.keras import Model
from keras.models import load_model
from tensorflow.keras import optimizers, layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.keras import callbacks

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.decomposition import PCA

def cap(var):  
    
    '''
    change la direction du vent en variable categorielle.
    '''
    
    if (var>337.5) or (var<=22.5):
        var=0
    elif (var>22.5) and (var<=67.5):
        var=45
    elif (var>67.5) and (var<=112.5):
        var=90
    elif (var>112.5) and (var<=157.5):
        var=135
    elif (var>157.5) and (var<=202.5):
        var=180
    elif (var>202.5) and (var<=247.5):
        var=225
    elif (var>247.5) and (var<=292.5):
        var=270
    elif (var>292.5) and (var<=337.5):
        var=315
        
    return var

''' importation des données station météo et passage de l index en datetime'''

data_station_meteo=pd.read_csv('/home/harry/station meteo/4 septembre/data_total.csv')
data_station_meteo.columns=['année', 'mois', 'jour', 'heure', 'minute', 'seconde', 'direction', 'force', 'humidité', 'température', 'pression','luminosité', 'pluie' ]
data_station_meteo=data_station_meteo.drop(['minute', 'seconde'], axis=1)
data_station_meteo['date']=data_station_meteo.apply(lambda row: str(int(row['jour']))+str('/')+str(int(row['mois']))+str('/')+str(int(row['année']))+str(' ')+str(int(row['heure']))+str(':')+str('0')+str(':')+str('0'), axis=1)
data_station_meteo['date']=pd.to_datetime(data_station_meteo['date'])
data_station_meteo=data_station_meteo.set_index('date')
data_station_meteo=data_station_meteo.drop(['année', 'jour', 'heure'], axis=1)
data_station_meteo['direction']=data_station_meteo['direction'].apply(lambda x: cap(x))
data_station_meteo=data_station_meteo.loc['2022-08-10 15:00:00':,:] # on garde les données pertinentes, les données d essai sont enlevées.



def traitement_image_temporel(chemin_data="/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_one/tenseur.npy", chemin_index="/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_one/index_tenseur.npy"):
    
    """
    fonction utile pour préparer les données prêtes à passer
    aux LSTM.On récupère le tenseur contenant les phots reformatées et  l index, 
    puis ce programme fournit le tenseur mais cette fois avec les images dans le bon ordre.
    """


    data_one=np.load(chemin_data)
    index_data_one=np.load(chemin_index)
    index_indices=[i for i in range(len(index_data_one))]

    total=pd.DataFrame({'index_data': index_indices, 'time': index_data_one})
    total['time']=total['time'].str.strip('_photo .jpg_')
    total['time']=total['time'].apply(lambda x: x.split('_'))
    total['time']=total['time'].apply(lambda x: x[0]+"/"+x[1]+"/"+x[2]+" "+x[3]+":"+str(0)+":"+str(0))
    total['time']=pd.to_datetime(total['time'])
    total=total.sort_values(by='time').reset_index()
    total['data']=total.apply(lambda row: np.expand_dims(data_one[row['index_data']], axis=0), axis=1)
    dede=total.data
    new_data=np.concatenate(total['data'], axis=0)
    
    return new_data, total.time
    
images, dates=traitement_image_temporel()

images_2, dates_2=traitement_image_temporel(chemin_data='/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_two/tenseur.npy', chemin_index='/home/harry/station meteo/analyse photos ciel/images_station_reformatées/64_64_NB_two/index_tenseur.npy')

''' vérification du bon agencement des images'''
'''
plt.figure(figsize=(20, 20))
plt.title("images redimensionnées jeu un")
for i in range(1, 21):
    plt.subplot(10, 2, i)
    plt.imshow(images[i])
    plt.xlabel(dates[i])
    plt.subplots_adjust(hspace=1.2)
    
plt.figure(figsize=(20, 20))
plt.title("images redimensionnées jeu deux")
for i in range(1, 21):
    plt.subplot(10, 2, i)
    plt.imshow(images_2[i])
    plt.xlabel(dates_2[i])
    plt.subplots_adjust(hspace=1.2)
'''   
''' importation de l encodeur et encodage des images remises dans  l ordre'''
    
encodeur=load_model('/home/harry/station meteo/analyse photos ciel/encodeur_convolutif.h5')

images_encodées=encodeur.predict(images)
images_encodées=images_encodées[:,0,0,:]
images_encodées=pd.DataFrame(index=dates, data=images_encodées)

images_encodées_2=encodeur.predict(images_2)
images_encodées_2=images_encodées_2[:,0,0,:]
images_encodées_2=pd.DataFrame(index=dates_2, data=images_encodées_2)


   

''' préparation des données pour l entrainement LSTM'''

def images_encodées_transform(images_encodées=images_encodées, cibles=data_station_meteo, days=3, freq_heure='H', prediction=1, var_prediction=['direction'], garbage=['pluie'], paquet='all', normaliser='oui', type_shape='classique'):
    
    '''
    cette fonction prépare les images encodées de la station pour le deep ou machine
    learning style LSTM.
    
    images_encodées: images à réagencer.
    
    cibles: données station météo pour utiliser comme cible à prédire.
    
    days: indique si on veut avoir les données agencées tout les jours, deux jours, trois jours...
    
    freq_heure: indique si on veut des données agencées toutes les deux heures, toutes les heures...
    si toutes les heures, H. si toutes les deux heures 2H. ENsuite 3H (trois heures) etc...
    
    prediction: prediction à un jour, deux jours...
    
    var_prediction: variables à prédire (direction, force, humidité, température, luminosité, pluie)
    
    paquet: permet de sélectionner soit juste les variables meteo ('variables'), soit juste les images ('images'), soit le tout ('all')
    
    type_shape: si on veut les données pretes pour les modeles classiques de machine ou deep learning ('classique') ou pour RNN ('RNN')
    
    
    '''
    cibles=cibles.drop(garbage, axis=1)
    
   
    
    images_encodées=cibles.join(images_encodées, how='outer')
    
    
    
    
    delta_dataset=dt.timedelta(days=days) - dt.timedelta(hours=1)
    delta_target=dt.timedelta(days=prediction)
    
    dataset=[]
    targets=[]
    dates=[]
    months=[]
    
    nan_dataset=[]
    nan_targets=[]
    nan_dates=[]
    
    for date in images_encodées.index:
        
        after=date+delta_target
        before=date-delta_dataset
        before_index=pd.date_range(start=before, end=date, freq=freq_heure)
        before_index=pd.DataFrame(before_index).set_index(0)
    
        features=before_index.join(images_encodées, how='left')
    
        try:
            
            
            target=images_encodées.loc[after, var_prediction]
            
        except:
            
            target=np.empty((1, len(var_prediction)))
            target=target.fill(np.nan)
            target=pd.DataFrame(target)
        
        if (features.isnull().values.any()==False) and (target.isnull().values.any()==False) and (target.shape[0]==len(var_prediction)):
            
            dataset.append(features)
            targets.append(np.array(target).reshape(1, len(var_prediction)))
            dates.append(after)
            
        else:
            nan_dataset.append(features)
            nan_targets.append(target)
            nan_dates.append(after)
           
        
    targets=np.concatenate(targets, axis=0)
    
    dataset_2=[]      
    for df in dataset:
        
        mois=df['mois'].unique().mean()
        df=df.drop('mois', axis=1)
        
        if paquet=='images':
            df=df.iloc[:,7-len(garbage):]
            df=np.array(df)
            if type_shape=='RNN':
                df=df.reshape(1, days*24, 960)
            elif type_shape=='classique':
                df=df.reshape(1, (days*24)*960)
        elif paquet=='all':
            df=df.iloc[:,:]
            df=np.array(df)
            if type_shape=='RNN':
                df=df.reshape(1, days*24, 960+7-len(garbage))
                
            elif type_shape=='classique':
                df=df.reshape(1, (days*24)*(960+7-len(garbage)))
                
        elif paquet=='variables':
            df=df.iloc[:,:7-len(garbage)]
            df=np.array(df)
            if type_shape=='RNN':
                df=df.reshape(1, days*24, 7-len(garbage))
                
            elif type_shape=='classique':
                df=df.reshape(1, (days*24)*(7-len(garbage)))
            
        if len(df.shape)==3 :
            mois=[mois for i in range((days*24)*(7-len(garbage)))]
            mois=np.array(mois).reshape(1, days*24, 7-len(garbage))
            df=np.concatenate((df, mois), axis=2)
            if paquet=='variables':
                df=df[:,:,:(7-len(garbage)+1)]
            elif paquet=='images':
                df=df[:,:,:961]
            elif paquet=='all':
                df=df[:,:,:960+7-len(garbage)+1]
        elif len(df.shape)==2:
            mois=np.array(mois).reshape(1, 1)
            df=np.concatenate((df, mois), axis=1)
        
        
        dataset_2.append(df)
        debug=dataset_2
    new_data=np.concatenate(dataset_2)
    
    return  new_data, targets, dates, months
    
data_1, cibles_1, dates_1, months_1=images_encodées_transform(days=3, paquet='variables', var_prediction=['luminosité', 'humidité', 'température', 'direction', 'force'], type_shape='RNN')
    
data_2, cibles_2, dates_2, months_2=images_encodées_transform(images_encodées=images_encodées_2, days=3, paquet='variables', var_prediction=['luminosité', 'humidité', 'température', 'direction', 'force'], type_shape='RNN')

data_total_var=np.concatenate([data_1, data_2])

cibles_total_var=np.concatenate([cibles_1, cibles_2])


data_1_img, cibles_1_img, dates_1_img, months_1_img=images_encodées_transform(days=3, paquet='images', var_prediction=['luminosité', 'humidité', 'température', 'direction', 'force'], type_shape='RNN')
    
data_2_img, cibles_2_img, dates_2_img, months_2_img=images_encodées_transform(images_encodées=images_encodées_2, days=3, paquet='images', var_prediction=['luminosité', 'humidité', 'température', 'direction', 'force'], type_shape='RNN')

data_total_img=np.concatenate([data_1_img, data_2_img])

cibles_total_img=np.concatenate([cibles_1_img, cibles_2_img])


''' preparation des données variables et images '''
'''
X_train=data_total_var[:3500,:,:]
X_test=data_total_var[3500:,:,:]
y_train=cibles_total_var[:3500,:]
y_test=cibles_total_var[3500:,:]

X_train_img=data_total_img[:3500,:,:-1]
X_test_img=data_total_img[3500:,:,:-1]
y_train_img=cibles_total_img[:3500,:]
y_test_img=cibles_total_img[3500:,:]

'''
X_train, X_test, y_train, y_test=train_test_split(data_total_var, cibles_total_var, test_size=0.2, random_state=35)

X_train_img, X_test_img, y_train_img, y_test_img=train_test_split(data_total_img, cibles_total_img, test_size=0.2, random_state=35)


''' modele RNN pour entrainer et prédire la variable selectionnée'''

target_encoder_direction=LabelEncoder() # pour la variable direction qui est categorielle.
target_encoder_direction.fit(y_train[:,3])
y_train[:,3]=target_encoder_direction.transform(y_train[:,3])
y_test[:,3]=target_encoder_direction.transform(y_test[:,3])

target_encoder_force=LabelEncoder() # pour la variable force qui est categorielle.
target_encoder_force.fit(y_train[:,4])
y_train[:,4]=target_encoder_force.transform(y_train[:,4])
y_test[:,4]=target_encoder_force.transform(y_test[:,4])

scaler_data=MinMaxScaler() # pour les autres variables.
scaler_data.fit(y_train[:,0:3])
y_train[:,0:3]=scaler_data.transform(y_train[:,0:3])
y_test[:,0:3]=scaler_data.transform(y_test[:,0:3])


for var in range(X_train.shape[2]): # pour normaliser les données en mode RNN seulement.
    scaler=MinMaxScaler()
    scaler.fit(X_train[:,:,var])
    X_train[:,:,var]=scaler.transform(X_train[:,:,var])
    X_test[:,:,var]=scaler.transform(X_test[:,:, var])


input_=layers.Input(shape=X_train.shape[1:])
input_img=layers.Input(shape=X_train_img.shape[1:])
hidden_img=layers.LSTM(200, return_sequences=True)(input_img)
hidden_img_2=layers.LSTM(100, return_sequences=True)(hidden_img)
hidden_img_3=layers.LSTM(50, return_sequences=True)(hidden_img_2)
hidden_img_4=layers.LSTM(2, return_sequences=True)(hidden_img_3)
concat=layers.concatenate([input_, hidden_img_3])
hidden_main=layers.LSTM(300, return_sequences=True)(concat)
hidden_main_2=layers.LSTM(200, return_sequences=True)(hidden_main)
hidden_luminosité=layers.LSTM(100)(hidden_main_2)
hidden_humidité=layers.LSTM(100)(hidden_main_2)
hidden_température=layers.LSTM(100)(hidden_main_2)
hidden_direction=layers.LSTM(100)(hidden_main_2)
hidden_force=layers.LSTM(100)(hidden_main_2)
hidden_luminosité_dropout=layers.Dropout(0.5)(hidden_luminosité)
hidden_humidité_dropout=layers.Dropout(0.5)(hidden_humidité)
hidden_température_dropout=layers.Dropout(0.5)(hidden_température)
hidden_direction_dropout=layers.Dropout(0.5)(hidden_direction)
hidden_force_dropout=layers.Dropout(0.5)(hidden_force)
output_luminosité=layers.Dense(1)(hidden_luminosité_dropout)
output_humidité=layers.Dense(1)(hidden_humidité_dropout)
output_température=layers.Dense(1)(hidden_température_dropout)
output_direction=layers.Dense(8, activation='softmax')(hidden_direction_dropout)
output_force=layers.Dense(6, activation='softmax')(hidden_force_dropout)


model=Model(inputs=[input_, input_img], outputs=[output_luminosité, output_humidité, output_température, output_direction, output_force])


print(model.summary())

loss_dict={'dense': 'mse', 'dense_1': 'mse', 'dense_2': 'mse', 'dense_3': 'sparse_categorical_crossentropy', 'dense_4': 'sparse_categorical_crossentropy'}

metrics_dict={'dense': 'mae', 'dense_1': 'mae', 'dense_2': 'mae', 'dense_3': 'accuracy', 'dense_4': 'accuracy'}

model.compile(loss=loss_dict, loss_weights=[0.2, 0.2, 0.2, 0.2, 0.2], metrics=metrics_dict, optimizer=optimizers.Adam(learning_rate=0.0007))

stop=callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=7)

history=model.fit([X_train, X_train_img], [y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3], y_train[:,4]], epochs=200, validation_data=([X_test, X_test_img], [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3], y_test[:,4]]), callbacks=stop)
'''
plt.title("réseau rnn mixte, luminosité, humidité, température.")
pd.DataFrame(history.history).plot(figsize=(30, 20))
plt.grid(True)
plt.gca().set_ylim(0, 2)
'''
visuel=pd.DataFrame(history.history)
plt.figure(figsize=(20, 20))
plt.subplot(2, 3, 1)
plt.title('luminosité')
plt.plot(visuel['dense_loss'], color='blue')
plt.plot(visuel['dense_mae'], color='red')
plt.plot(visuel['val_dense_loss'], color='orange')
plt.plot(visuel['val_dense_mae'], color='green')
plt.subplot(2, 3, 2)
plt.title('humidité')
plt.plot(visuel['dense_1_loss'], color='blue')
plt.plot(visuel['dense_1_mae'], color='red')
plt.plot(visuel['val_dense_1_loss'], color='orange')
plt.plot(visuel['val_dense_1_mae'], color='green')
plt.subplot(2, 3, 3)
plt.title('température')
plt.plot(visuel['dense_2_loss'], color='blue')
plt.plot(visuel['dense_2_mae'], color='red')
plt.plot(visuel['val_dense_2_loss'], color='orange')
plt.plot(visuel['val_dense_2_mae'], color='green')
plt.subplot(2, 3, 4)
plt.title('direction')
plt.plot(visuel['dense_3_loss'], color='blue')
plt.plot(visuel['dense_3_accuracy'], color='red')
plt.plot(visuel['val_dense_3_loss'], color='orange')
plt.plot(visuel['val_dense_3_accuracy'], color='green')
plt.subplot(2, 3, 5)
plt.title('force')
plt.plot(visuel['dense_4_loss'], color='blue')
plt.plot(visuel['dense_4_accuracy'], color='red')
plt.plot(visuel['val_dense_4_loss'], color='orange')
plt.plot(visuel['val_dense_4_accuracy'], color='green')
plt.show()

print(model.evaluate([X_test, X_test_img], [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3], y_test[:,4]]))




