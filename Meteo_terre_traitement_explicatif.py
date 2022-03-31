# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:16:02 2021

@author: Utilisateur
"""


""" We use those packages. """

from datetime import datetime
import numpy as np
import pandas as pd

""" This function is useful to turn wind direction variable into categorical."""

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


""" This next function convert wind strength in knots and turn this variable into categorical."""

def vent(var):
    
    var=var*3600//1852
    
    if (var>=0) and (var<5):
        var=1
    
    elif (var>=5) and (var<10):
        var=2
        
    elif (var>=10) and (var<15):
        var=3
        
    elif (var>=15) and (var<20):
        var=4
        
    elif (var>=20) and (var<25):
        var=5
        
    elif (var>=25) and (var<30):
        var=6
        
    else:
        var=7
        
    return var


""" This another function takes care of rainfall and turn it into binary variable."""

def flotte(var):
    
    if var==0:
        var=0
        
    elif var!=0:
        var=1
        
    return var

""" This one is used to convert temperature from Kelvin to Celsius, and turn this variable into categorical."""
        
def glagla(var):
    
    var=var-273.15
    
    if var<0:
        var=0
        
    elif (var>=0) and (var<5):
        var=1
        
    elif (var>=5) and (var<10):
        var=2
        
    elif (var>=10) and (var<15):
        var=3
        
    elif (var>=15) and (var<20):
        var=4
        
    elif (var>=20) and (var<25):
        var=5
        
    elif (var>=25) and (var<30):
        var=6
        
    elif var>=30:
        var=7
        
    return var

""" This function takes care of sunlight intensity variable, and make it categorical."""

def sun(var):
    
    if var==0:
        var=0
        
    elif (var>0) and (var<=50):
        var=1
        
    elif (var>50) and (var<=100):
        var=2
        
    elif (var>100) and (var<=150):
        var=3
        
    elif (var>150) and (var<=200):
        var=4
        
    elif (var>200) and (var<=250):
        var=5
        
    elif (var>250) and (var<=300):
        var=6
        
    elif (var>300) and (var<=350):
        var=7
        
    elif (var>350) and (var<=400):
        var=8
        
    elif (var>400) and (var<=450):
        var=9
        
    elif (var>450) and (var<=500):
        var=10
        
    elif (var>500) and (var<=550):
        var=11
        
    elif (var>550) and (var<=600):
        var=12
        
    elif (var>600) and (var<=650):
        var=13
        
    elif (var>650) and (var<=700):
        var=14
        
    elif (var>700) and (var<=750):
        var=15
        
    elif (var>750) and (var<=800):
        var=16
        
    elif (var>800) and (var<=850):
        var=17
        
    elif (var>850) and (var<=900):
        var=18
        
    elif (var>900) and (var<=950):
        var=19
        
    elif (var>950) and (var<=1000):
        var=20
        
    elif (var>1000) and (var<=1050):
        var=21
        
    elif (var>1050) and (var<=1100):
        var=22
        
    elif (var>1100) and (var<=1150):
        var=23
        
    elif (var>1150) and (var<=1200):
        var=24
        
    elif (var>1200) and (var<=1250):
        var=25
        
    elif (var>1250) and (var<=1300):
        var=26
        
    elif (var>1300) and (var<=1350):
        var=27
        
    elif (var>1350) and (var<=1400):
        var=28
        
    elif (var>1400) and (var<=1450):
        var=29
        
    elif (var>1450) and (var<=1500):
        var=30
        
    elif (var>1500) and (var<=1550):
        var=31
        
    elif (var>1550) and (var<=1600):
        var=32
        
    elif (var>1600) and (var<=1650):
        var=33
        
    else:
        var=34
        
    return var

""" This class create an object able to organise data from MeteoNet, ready for sklearn (see the MTTT notice for more accuracy). """


class Meteo_Data_View:
    
    
    def __init__(self, name, heures=1, jours=1, var_corbeille=[], vue=12, cible='direction'):
        self.name=name
        numb_vars=8 # indicates only variables changing per hour.
        
        stations_meteo=[] # each stations_meteo list will help us to treat each station data.
        stations_meteo_2=[]
        stations_meteo_3=[]
        
        data_good_shape=[] # this will be the final prepared dataset.
        target=[] # this will be the final prepared label list.
        compteur=[] # it will be used to select stations and avoid being blocked in a loop.
        
        var_soleil=pd.read_csv('var_soleil_région.csv') # on this paragraph we import the sunlight data created before in 'france_regions_frontières'. The purpose is to get a table with sunlight intensity per hour and per France areas (8 areas here, one column per area, and rows are hours).
        var_soleil=var_soleil.set_index('temps')
        var_soleil.index=pd.to_datetime(var_soleil.index)
        var_soleil=var_soleil.resample(str(heures)+'H').mean()
        régions_barycentres=pd.read_csv('régions_barycentres.csv') # here we import a table where we have the coordinates centroid for each area.
        var_soleil.columns=['Bourgogne Franche-Comté', 'Aquitaine Limousin Poitou-Charentes', 'Nord-Pas-de-Calais Picardie', 'Auvergne Rhône-Alpes', 'Bretagne', 'Languedoc-Roussillon Midi-Pyrénées', 'Pays de la Loire', 'Centre-Val de Loire', 'Alsace Champagne-Ardenne Lorraine', 'Provence-Alpes-Côte d\'Azur', 'Normandie', 'Île-de-France']
        
        
        indexage_heures=list(pd.date_range('2016-01-01 00:00:00', '2018-12-31 23:00:00', freq=str(heures)+'h')) # we write a datetime index on hours.
        time_heures=pd.DataFrame({'temps': indexage_heures}) 
        time_heures=time_heures.set_index('temps') 
        
        indexage_jours=list(pd.date_range('2016-01-01', '2018-12-31', freq='d')) # the same but now on days.
        time_jours=pd.DataFrame({'temps': indexage_jours})
        
        if cible=='direction': # 'ci' will be useful to fill up the 'target' list a little further.
            ci=3
        elif cible=='force':
            ci=4
        elif cible=='pluie':
            ci=5
        elif cible=='température':
            ci=8
        elif cible=='humidité':
            ci=6
        elif cible=="pression":
            ci=9
        elif cible=='point_rosée':
            ci=7
        elif cible=='soleil':
            ci=10
    
        for station in self.name['number_sta']: # here for each station we organise data in one hour per row in the good time sense without any hole, and we turn it in 'int' to use less power without losing information.
            
           station_data=self.name.loc[self.name['number_sta']==station]
           station_data=station_data.sort_values(['date'],ascending=True)
           station_data=station_data.set_index('date')
           station_data.index=pd.to_datetime(station_data.index)
           station_data=station_data.resample(str(heures)+'H').mean()
           station_data=time_heures.join(station_data, how='outer')
           station_data["dd"]=station_data["dd"].map(lambda x: cap(x))
              
           
            
           station_data['number_sta']=station_data['number_sta'].apply(lambda x: int(x) if np.isnan(x)==False else x)
           station_data['lat']=station_data['lat'].apply(lambda x: int(x*100) if np.isnan(x)==False else x)
           station_data['lon']=station_data['lon'].apply(lambda x:int(x*100) if np.isnan(x)==False else x)
           station_data['height_sta']=station_data['height_sta'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           station_data['dd']=station_data['dd'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           station_data['ff']=station_data['ff'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['hu']=station_data['hu'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['precip']=station_data['precip'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['psl']=station_data['psl'].apply(lambda x:int(x/100) if np.isnan(x)==False else x)
           station_data['td']=station_data['td'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['t']=station_data['t'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           
           if station_data['number_sta'].mean() not in compteur:
               
              stations_meteo.append(station_data)
           
           else: break
       
           compteur.append(station_data['number_sta'].mean())
        
                
        for station_data in stations_meteo: # here we get rid of 'nan' values. For a 'nan', the row is cancelled.
            
            station_data=station_data.dropna(axis=1, how='all')
            station_data=station_data.dropna(axis=0, how='any')
            station_data=time_heures.join(station_data, how='outer')
            
            
            distances_latitudes_barycentres=[] # from here to 'station_data', we join the sunlight intensity variable on each sample. For that we calculate the distance from the sample to each centroid, and we keep data of the closest one.
            distances_longitudes_barycentres=[]
            noms_régions=[]
        
            for latitude in régions_barycentres['latitude']:
               dis_lat=latitude*100-station_data['lat'][0]
               distances_latitudes_barycentres.append(dis_lat)
               noms_régions.append(régions_barycentres[régions_barycentres['latitude']==latitude].régions)
            for longitude in régions_barycentres['longitude']:
               dis_lon=longitude*100-station_data['lon'][0]
               distances_longitudes_barycentres.append(dis_lon)
          
            distances_barycentres=[np.sqrt(i**2+j**2) for i,j in zip(distances_latitudes_barycentres, distances_longitudes_barycentres)]
            distances_régions_barycentres=pd.DataFrame({'région': noms_régions, 'distance': distances_barycentres})
            distances_régions_barycentres=distances_régions_barycentres.sort_values(['distance'], ascending=True)
            distances_régions_barycentres=distances_régions_barycentres.reset_index()
            
            soleil=var_soleil[distances_régions_barycentres['région'][0]]
            
            
            station_data=station_data.join(soleil, how='left')
            
            
            
            if (station_data.shape[1]==12): # if no dimension/variable is missing in the station, we keep it.
                
                stations_meteo_2.append(station_data)
        
          
        
        
        
        for station_data in stations_meteo_2:
            # in this paragraph, we organise data in a day for a sample.
            liste_heures=[station_data[station_data.index.hour==i] for i in range(0,24,heures)]
            station_data=np.concatenate(liste_heures,axis=1)
            station_data=pd.DataFrame(station_data)
            station_data.columns=list(np.arange(0, station_data.shape[1]))
            
            # here we drop motionless variables (number_sta, latitude, longitude, height) who pop up several times per sample.In fact we drop occurences except for number_sta,where we drop all.
            station_data=station_data.drop([0+(i*12) for i in range(int(24/heures))],axis=1) 
            station_data=station_data.drop([13+(i*12) for i in range(int(23/heures))],axis=1) 
            station_data=station_data.drop([14+(i*12) for i in range(int(23/heures))],axis=1) 
            station_data=station_data.drop([15+(i*12) for i in range(int(23/heures))],axis=1) 
            # now we organise data with a day per sample and without any hole.
            station_data=time_jours.join(station_data)
            station_data=station_data.set_index('temps')
            station_data['mois']=station_data.index.month # we add up a variable 'mois', indicate the month of the sample (like a season indicator per row).
            
            # here we organise data per day or days per row.
            liste_jours=[station_data[i:-jours+i:jours] for i in range(jours)]      
            station_data=np.concatenate(liste_jours, axis=1)
            station_data=pd.DataFrame(station_data)
            
            
            
            # once again, this paragraph drops occurences of latitude, longitude, mois if we have several days per sample.
                
            lat=[(((24/heures)*numb_vars+4)*n) for n in range(1,jours)]
            lon=[(((24/heures)*numb_vars+4)*n)+1 for n in range(1,jours)]
            hei=[(((24/heures)*numb_vars+4)*n)+2 for n in range(1,jours)]
            mon=[((24/heures)*numb_vars+4)*n-1 for n in range(1,jours)]
                
            station_data=station_data.drop(lat, axis=1)
            station_data=station_data.drop(lon, axis=1)
            station_data=station_data.drop(hei, axis=1)
            station_data=station_data.drop(mon, axis=1)
                
            
            
            station_data.columns=list(np.arange(0, station_data.shape[1]))
            
            # here we will create new samples by using a hours shift.
            
            liste_decalage_heures_station=[]
            
            
            for decalage in range(int(24/heures)*jours):
                
                part_1=station_data.iloc[0:(station_data.shape[0]-1),0:3].reset_index(drop=True)
                part_2=station_data.iloc[0:(station_data.shape[0]-1),3+numb_vars*decalage:-1].reset_index(drop=True)
                part_3=station_data.iloc[1:(station_data.shape[0]),3:3+numb_vars*decalage].reset_index(drop=True)
                part_4=station_data.iloc[1:(station_data.shape[0]),-1].reset_index(drop=True)
                total=part_1.join(part_2)
                total=total.join(part_3)
                total=total.join(part_4)
                total.columns=list(np.arange(0,total.shape[1]))
                liste_decalage_heures_station.append(total)
                
            stations_meteo_3.append(liste_decalage_heures_station)
        # now for each row we will keep only full rows without 'nan' in the dataset and their linked target without 'nan' too.  
      
        for station_data in stations_meteo_3:
            
            for station_decalage in station_data:
                
                station_decalage=station_decalage.reset_index(drop=True)
                                  
                for val in range(station_decalage.shape[0]-1):
                        
                    row=station_decalage.iloc[val,:]
                    row2=station_decalage.iloc[val+1,numb_vars*heures*(jours-1)+ci+numb_vars*vue]
                    
                    
                    if (np.isnan(row).sum()==0) and (np.isnan(row2).sum()==0):
                        
                        data_good_shape.append(row)
                        target.append(row2)
                            
                        
                            
                    
                    else:
                        continue
        
        data_good_shape=pd.DataFrame(data_good_shape)
        # here we will drop unneeded variables for us if we want to delete some.
        for name in var_corbeille:
                              
                if name.find("force")!=-1:
                    data_good_shape=data_good_shape.drop([4+i*numb_vars for i in range(0,jours*24)], axis=1)
                
                elif name.find("direction")!=-1:
                    data_good_shape=data_good_shape.drop([3+i*numb_vars for i in range(0,jours*24)], axis=1)
                  
                elif name.find("pluie")!=-1:
                    data_good_shape=data_good_shape.drop([5+i*numb_vars for i in range(0, jours*24)], axis=1)
                    
                elif name.find("humidité")!=-1:
                    data_good_shape=data_good_shape.drop([6+i*numb_vars for i in range(0,jours*24)], axis=1)
                    
                elif name.find("point_rosée")!=-1:
                    data_good_shape=data_good_shape.drop([7+i*numb_vars for i in range(0,jours*24)], axis=1)
                    
                elif name.find("température")!=-1:
                    data_good_shape=data_good_shape.drop([8+i*numb_vars for i in range(0,jours*24)], axis=1)
                    
                elif name.find("pression")!=-1:
                    data_good_shape=data_good_shape.drop([9+i*numb_vars for i in range(0,jours*24)], axis=1)
                 
                elif name.find("soleil")!=-1:
                    data_good_shape=data_good_shape.drop([10+i*numb_vars for i in range(0, jours*24)], axis=1)
                 
                elif name.find("latitude")!=-1:
                    data_good_shape=data_good_shape.drop([0], axis=1)
                    
                elif name.find("longitude")!=-1:
                    data_good_shape=data_good_shape.drop([1], axis=1)
                    
                elif name.find("temps")!=-1:
                    data_good_shape=data_good_shape.drop([3+24*numb_vars*jours], axis=1)
        
                elif name.find("altitude")!=-1:
                    data_good_shape=data_good_shape.drop([2], axis=1)
        
            
                    
        target=np.transpose(target) 
        data_good_shape=np.array(data_good_shape)
        # here we use our functions to get our categories.
        target_2=[]
        if cible=='force':
            for val in target:
                val2=vent(val)
                target_2.append(val2)
            target=target_2
        elif cible=='pluie':
            for val in target:
                val2=flotte(val)
                target_2.append(val2)
            target=target_2
            
        elif cible=='température':
            for val in target:
                val2=val
                target_2.append(val2)
            target=target_2
            
        elif cible=='soleil':
            for val in target:
                val2=val
                target_2.append(val2)
            target=target_2
        
        
        target=np.array(target)
        target=target.astype('int32')
        data_good_shape=data_good_shape.astype('int32')
        
       
        # with our object we can have the dimensions of it, prepared dataset and labels, the sunlight intensity table for areas in France, and the centroids table for areas in france.
        
        self.dimensions=data_good_shape.shape
        self.target=target   
        self.data_good_shape=data_good_shape
        self.var_soleil=var_soleil
        self.régions_barycentres=régions_barycentres
        
        
        
        
       
      
    
    
    
    
    
    