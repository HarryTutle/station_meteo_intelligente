''' librairies indispensables pour la station meteo'''

import numpy as np # pour les manips de matrices.
import pandas as pd # idem mais pour des matrices indexées on va dire.
import csv # permet de manipuler les fichiers csv.
import serial # permet la communication entre la arduino uno et la raspberry pi.
import termios # permet d'éviter le reset de l'arduino quand on active la liaison série dans le script python.
import datetime # utile pour manipuler des données temporelles.
from scipy import stats

# import struct # utile pour envoyer les infos de la raspberry à l'arduino sous forme de structure.
import joblib # utile pour enregistrer des modèles sklearn, des array...
import sklearn # pour le machine learning et le prétraitement des données.
import tensorflow # pour le deep learning.
from tensorflow import keras # API de tensorflow, pratique pour le deep learning avec fonctions "clés en main".

from luma.core.interface.serial import i2c # luma sert à interfacer l'écran oled de la raspberry qui affichera les prévisions météo.
from luma.core.render import canvas 
from luma.oled.device import sh1106
import time # pour crée des délais d'affichage ou dans les boucles notamment, histoire d'avoir le temps de lire les résultats, faire des calculs...
import os # communiquer avec le terminal.
import sys # communiquer avec le terminal.

serial_ecran=i2c(port=1, address=0x3C) # initialisation adresse I2C de l'écran.
device=sh1106(serial_ecran)
device.persist= True;

''' fonction pour calibrer les données direction comme le set d'entrainement'''

def cap(var):  
    
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


''' lecture du port serie pour communiquer avec arduino et compilation dans un array '''

port="/dev/ttyACM0" # liaison port usb.
f=open(port)
attrs=termios.tcgetattr(f)
attrs[2]=attrs[2] & ~termios.HUPCL
termios.tcsetattr(f, termios.TCSAFLUSH, attrs)
f.close()
reception=serial.Serial()
reception.baudrate=115200
reception.timeout=5
reception.port=port

reception.open()


data=[]
min=0

while (min<4):  # temps que l'on reste en dessous des 4 minutes, on enregistre dans data ce qu'on reçoit.
    
    data_heure=str(reception.readline())
    data_heure=data_heure.strip("b'\\r\\n") # on vire les caractères non chiffrés.
    data_heure=data_heure.split() # on sépare les caractères pour chaque ligne.
    # print(data_heure)
    if (len(data_heure)==13): # dans data on ne garde que les échantillons chiffrés, on vire les strings inutiles des messages.
        min=int(data_heure[4]) # sort de la boucle quand on dépasse les 4 minutes.
        data.append(data_heure)
    

reception.close() 
data=np.array(data) # on convertit data en array d'entiers.
data=data.astype('int')


''' calcul de la valeur moyenne des données sur l'array '''


data=np.mean(data, axis=0)# on garde la valeur moyenne sur quelques minutes pour avoir une vision plus globale des observations (si rafales de vent, nuage qui passe...).
data=data.astype('int')


''' enregistrement de l'echantillon dans le fichier csv data_total sur la clé USB '''

fichier=open('/media/pi/UBUNTU 20_0/data_total.csv', 'a') # on enregistre sur clé usb pour soulager la durée de vie de la carte sd et aussi pour récupérer les données lors des phases de réentrainement.
objet=csv.writer(fichier)
objet.writerow(data)
fichier.close()
    

''' enregistrement d'une photo du ciel dans le fichier photo_data_total sur la clé USB '''


os.system(f"raspistill -o /media/pi/'UBUNTU 20_0'/photo_data_total/photo_{data[0]}_{data[1]}_{data[2]}_{data[3]}.jpg")


''' calcul des prévisions si les données sont suffisantes '''

# preparation des donnees enregistrees pour le modele predictif.

data_lieu=[4935, 97, 50] # coordonnées latitude, longitude et altitude de la station meteo.

data_prevision=pd.read_csv('/media/pi/UBUNTU 20_0/data_total.csv')


if (data_prevision.shape[0]>=72): # si on a collecté 3 jours de données par heure au moins, alors on peut faire une prévision.
    # là on va calculer la durée de l'échantillon (vérifie si il n y a pas eu de coupure dans la collecte des données sur les 72 heures. )
    debut_echantillon=datetime.datetime(year=int(data_prevision.iloc[-73, 0]), month=int(data_prevision.iloc[-73, 1]), day=int(data_prevision.iloc[-73, 2]), hour=int(data_prevision.iloc[-73, 3]))
    fin_echantillon=datetime.datetime(year=int(data_prevision.iloc[-1, 0]), month=int(data_prevision.iloc[-1, 1]), day=int(data_prevision.iloc[-1, 2]), hour=int(data_prevision.iloc[-1, 3]))
    duree_mesure_meteo=fin_echantillon-debut_echantillon
    duree_jours_mesure_meteo=int(duree_mesure_meteo.days)
    duree_secondes_mesure_meteo=int(duree_mesure_meteo.seconds)
    
    
    mois=[data_prevision.iloc[-1, 1]] # on isole le mois que l'on mettra à la fin ensuite, comme le format de nos données d'entrainement.
    data_prevision=data_prevision.drop(['0.0.1'], axis=1) # on enlève le paramètre pluie pour notre prévision.
    data_prevision['0.0']=data_prevision['0.0'].apply(lambda x:int(x*1852/3600)) # on reconvertit le vent en m.s comme les données d'entrainement
    data_prevision['180.0']=data_prevision['180.0'].apply(lambda x:int(cap(x))) # on utilise la fonction cap pour recalibrer correctement la direction du vent pour la prévision.
    data_prevision['14.933333333333334']=data_prevision['14.933333333333334'].apply(lambda x:int(x*800/1024)) # on convertit les mesures du phototransistor à la même échelle que les données de ECO2MIX où la valeu max enregistrée est de 1470. La borne analogique de l'arduino va potentiellement jusqu'à 1024.
    data_prevision['20.0']=data_prevision['20.0'].apply(lambda x:int(273.15+x)) # on convertit les celsius en kelvin, comme le format des données d'entrainement.
    data_temps=data_prevision.iloc[-1, :6] # heure de la dernière mesure avant prevision.
    data_prevision=data_prevision.iloc[-72:,6:] # on garde seulement les 72 dernières heures pour la future prévision et on vire les variables temps.
    data_prevision.columns=list(range(data_prevision.shape[1])) # ces deux commandes permettent de réinitiliser l'indexation du dataframe pour mieux le modifier ensuite.
    data_prevision=data_prevision.reset_index(drop=True)
    
    
    
    if (duree_jours_mesure_meteo != 3): # si la durée de l'échantillon dépasse 3 jours ou en plus 2 heures (on se laisse une petite marge erreur), alors la prédiction sera corrompue donc on recommence la collecte des données.
        
        
        
        with canvas(device) as draw:
    
            draw.text((0, 0), str("trou dans les"), fill="white")
      
            draw.text((0, 10), str("données..."), fill="white")
      
            draw.text((0, 20), str("recollecte de"), fill="white")
      
            draw.text((0, 30), str(f"données."), fill="white")
      
    
        time.sleep(20) # permet 20 secondes de pause lecture ecran.
        
      
        sending=[data[3], data[6], data[7], data[8], data[9], data[10], data[11], data[12], 0, 0, 0, 0, 0]
                                                                                                                                                                                  
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    
        data_prevision=[data_prevision.iloc[i,:] for i in range(72)] 
        data_prevision=pd.concat(data_prevision, axis=0)
        data_prevision=pd.concat([pd.DataFrame(data_lieu), data_prevision, pd.DataFrame(mois)], axis=0)
        data_prevision=data_prevision.T
        data_prevision=np.array(data_prevision) # ces dernières manips permettent d'agencer notre dataframe en array sur une seule ligne, en séquence.
        
   
        # direction du vent

        model=keras.models.load_model('/home/pi/Desktop/direction 24h layers/layers_direction_24h.h5', compile=False) # importation du modèle direction vent.
        prepro=joblib.load('/home/pi/Desktop/direction 24h layers/preprocessor_direction_24h.joblib') # importation du modèle pour prétraiter l'échantillon avant la prévision.
        encoder=joblib.load('/home/pi/Desktop/direction 24h layers/encodeur_direction_24h.joblib') # importation du décodeur des labels.
        data_prevision_preprocessed=prepro.transform(data_prevision) # on prétraite l'échantillon.
        decoder=[]
        for val in model.predict(data_prevision_preprocessed): # puis on prédit la direction du vent avec le modèle.
            decoder.append(np.argmax(val))
        vent_prevision=encoder.inverse_transform(decoder) # on retrouve la prévision en sélectionnant la proba maximum entre les résultats possibles.
        vent_prevision=int(vent_prevision)
        
    
        # force du vent
    
        model=keras.models.load_model('/home/pi/Desktop/force 24h layers/layers_force_24h.h5', compile=False)
        prepro=joblib.load('/home/pi/Desktop/force 24h layers/preprocessor_force_24h.joblib')
        encoder=joblib.load('/home/pi/Desktop/force 24h layers/encodeur_force_24h.joblib')
        data_prevision_preprocessed=prepro.transform(data_prevision)
        decoder=[]
        for val in model.predict(data_prevision_preprocessed):
            decoder.append(np.argmax(val))
        force_prevision=encoder.inverse_transform(decoder)
        force_prevision=int(force_prevision)
    
        if force_prevision==0:  # la boucle if permet de relier le label aux forces de vent du modèle d'entrainement.
            
            force_prevision_2=str("0")
            
        elif force_prevision==1:
            
            force_prevision_2=str("0/5")
            
        elif force_prevision==2:
            
            force_prevision_2=str("5/10")
            
        elif force_prevision==3:
            
            force_prevision_2=str("10/15")
            
        elif force_prevision==4:
            
            force_prevision_2=str("15/20")
            
        elif force_prevision==5:
            
            force_prevision_2=str("20/25")
            
        elif force_prevision==6:
            
            force_prevision_2=str("25/30")
            
        elif force_prevision==7:
            
            force_prevision_2=str("30+")
        
        
    
        # ensoleillement
    
        model=keras.models.load_model('/home/pi/Desktop/soleil 24h layers/layers_soleil_24h.h5', compile=False)
        prepro=joblib.load('/home/pi/Desktop/soleil 24h layers/preprocessor_soleil_24h.joblib')
        data_prevision_preprocessed=prepro.transform(data_prevision)
        soleil_prevision=model.predict(data_prevision_preprocessed)
        soleil_prevision=int(soleil_prevision)
        
    
        # humidité
    
        model=keras.models.load_model('/home/pi/Desktop/humidite 24h layers/layers_humidite_24h.h5', compile=False)
        prepro=joblib.load('/home/pi/Desktop/humidite 24h layers/preprocessor_humidite_24h.joblib')
        data_prevision_preprocessed=prepro.transform(data_prevision)
        humidite_prevision=model.predict(data_prevision_preprocessed)
        humidite_prevision=int(humidite_prevision)
        
    
        # temperature
    
        model=keras.models.load_model('/home/pi/Desktop/température 24h layers/layers_temperature_24h.h5', compile=False)
        prepro=joblib.load('/home/pi/Desktop/température 24h layers/preprocessor_temperature_24h.joblib')
        data_prevision_preprocessed=prepro.transform(data_prevision)
        temperature_prevision=model.predict(data_prevision_preprocessed)
        temperature_prevision=int(temperature_prevision)
        temperature_prevision=temperature_prevision-273
        
        
        
     
        ''' enregistrement de la prediction dans le fichier csv predict_total sur la clé USB '''

        predict_line=[data_temps[0], data_temps[1], data_temps[2]+1, data_temps[3], vent_prevision, force_prevision_2, soleil_prevision, humidite_prevision, temperature_prevision]
        predict_line=np.array(predict_line)
        fichier_2=open('/media/pi/UBUNTU 20_0/predict_total.csv', 'a')
        objet_2=csv.writer(fichier_2)
        objet_2.writerow(predict_line)
        fichier_2.close()
        
         
  
        with canvas(device) as draw:
    
            draw.text((0, 0), str("direction:"), fill="white")
            draw.text((60, 0), str(f"{vent_prevision} deg"), fill="white")
            draw.text((0, 10), str("force:"), fill="white")
            draw.text((60, 10), str(f"{force_prevision_2} knds"), fill="white")
            draw.text((0, 20), str("humidité:"), fill="white")
            draw.text((60, 20), str(f"{humidite_prevision} %"), fill="white")
            draw.text((0, 30), str("température:"), fill="white")
            draw.text((80, 30), str(f"{temperature_prevision} °C"), fill="white")
            draw.text((0, 40), str("soleil:"), fill="white")
            draw.text((60, 40), str(f"{soleil_prevision}"), fill="white")
    
        time.sleep(20) # permet 2 minutes de pause du programme pour lire l'écran avant extinction de la raspberry pi.

        
        sending=[data[3], data[6], data[7], data[8], data[9], data[10], data[11], data[12], vent_prevision, force_prevision, humidite_prevision, temperature_prevision, soleil_prevision]
        

else: # si on a pas assez de collectes dans notre fichier csv, on affiche le délai d'attente supplémentaire.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    
    with canvas(device) as draw:
    
        draw.text((0, 0), str("attente des"), fill="white")
      
        draw.text((0, 10), str("données, prévisions"), fill="white")
      
        draw.text((0, 20), str("possibles dans"), fill="white")
      
        draw.text((0, 30), str(f"{72-data_prevision.shape[0]} heures"), fill="white")
      
    
    time.sleep(20) # permet 20 secondes de pause du programme pour lire l'écran avant extinction de la raspberry pi.
  
    
    sending=[data[3], data[6], data[7], data[8], data[9], data[10], data[11], data[12], 0, 0, 0, 0, 0]

switcher_start=1 # va indiquer au récepteur le début de la transmission radio.
switcher_end=2 # va indiquer au récepteur la fin de la transmission.

os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_start}") # on envoie trois fois ce signal pour être sûr d'avoir une réception complète.
time.sleep(3)
os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_start}")
time.sleep(3)
os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_start}")
time.sleep(3)

for val in range(len(sending)):
   
   if sending[val]==0:
       sending[val]=5555 # la valeur zéro n'est pas transmise, je sais pas pourquoi... du coup le zero c'est 5555 et pi c'est tout, on le convertira au récepteur radio pour contourner.
       
   sending[val]=sending[val]+10000*(val+1)
   
   os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {sending[val]}")
   time.sleep(3) # petite pause de trois secondes entre chaque nombre transmit.
   
   os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {sending[val]}")
   time.sleep(3) # petite pause de trois secondes entre chaque nombre transmit. On le fait deux fois en cas de loupé.
   
os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_end}") # idem trois fois pour bien clôturer le message.
time.sleep(3)
os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_end}")
time.sleep(3)
os.system(f"cd /home/pi/Desktop/librairies/433Utils-master/RPi_utils && sudo ./codesend {switcher_end}")
time.sleep(3)

   







os.system("sudo shutdown -h now") # extinction des feux !
