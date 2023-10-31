#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:06:49 2023

@author: harry
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import glob
import os
import shutil
import re

def reformateur_image(chemin="", destination="", nouveau_format=[256, 256], couleur="NB", tenseur="oui"):
    """
    cette fonction reformate les images pour les enregistrer 
    dans un dossier. Renvoie aussi une liste avec les images 
    qui n ont pas pu etre traitées si il y en a, et renvoie un tenseur
    avec les images traitées pour keras si besoin.
    
    .chemin: indique en string le chemin absolu d'origine des images.
    
    .destination: indique l'emplacement et le nom du nouveau dossier 
    où sont enregistré les images reformatées.
    
    .nouveau_format: liste avec deux éléments à renseigner: hauteur et largeur
    de l image en pixels.Defaut [256, 256]
    
    .couleur: si on veut en couleur, 'coloré', sinon 'NB' pour noir et blanc(defaut NB).
    
    .tenseur: si 'oui' renvoie(defaut), si 'non' renvoie pas.
    """
    
    try:
        os.makedirs(destination)
    except:
        print("erreur chemin dossier ou dossier déjà crée.")
    
    
    garbage=[]
    try:
        for image in glob.glob(chemin+"/*"):
            
            new_img=Image.open(image)
            new_img=new_img.resize(nouveau_format)
            if couleur=="NB":
                new_img=new_img.convert('L')
            elif couleur=='coloré':
                pass
            new_img.save(destination+"/"+re.search('photo_[\d_]*.jpg', image).group(0))
    except:
        garbage.append(image)
        pass
    
    
    if tenseur=='oui':
        tenseur=[]
        index_tenseur=[]
        for image in glob.glob(destination+"/*"):
            #img_name=image[-17:]
            img=Image.open(image)
            img=np.asarray(img)
            tenseur.append(img)
            index_tenseur.append(re.search('photo_[\d_]*.jpg', image).group(0))
        if couleur=='NB':
            tenseur=np.array(tenseur)/255.0
            index_tenseur=np.array(index_tenseur)
        elif couleur=='coloré':
            tenseur=np.array(tenseur)
            index_tenseur=np.array(index_tenseur)
        np.save(destination+'/tenseur.npy', tenseur)
        np.save(destination+'/index_tenseur.npy', index_tenseur)
        
        return tenseur, index_tenseur, garbage
            
    elif tenseur=='non':
        
        return garbage
    
    





"""
a, b, c=reformateur_image(chemin="/home/harry/station meteo/30 juillet/photo_data_total"
                          , destination="/home/harry/station meteo/30 juillet/photo_data_total/photos_traitées_64*64_NB"
                          , nouveau_format=[64, 64]
                          , couleur="NB"
                          , tenseur='oui')
"""
a, b, c=reformateur_image(chemin="/home/harry/station meteo/30 juillet/photo_data_total"
                          , destination="/home/harry/station meteo/analyse photos ciel/images_station_reformatées/two"
                          , nouveau_format=[64, 64]
                          , couleur="NB"
                          , tenseur='oui')
