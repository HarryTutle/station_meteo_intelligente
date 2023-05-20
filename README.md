# station_meteo_intelligente
 les ingrédients pour fabriquer une station meteo 
 
 Actuellement le projet est dans mon jardin et tourne depuis huit mois. La station météo actuelle tourne en enregistrant la météo locale (direction, force, humidité, pluviométrie, température, pression et photo du ciel) toute les heures grâce à une carte arduino uno qui collecte via des composants (horloge externe ds1302, capteur bme280, phototransistor, anémo, girouette) les données pour les envoyer à une raspberry pi 3 B+ chargée d'enregistrer les données sur clé usb, prendre une photo du ciel et calculer la prévision pour les prochaines 24 heures. Enfin, elle envoie via un signal radio la météo actuelle et la prévision à un récepteur qui tourne avec une carte arduino simple.
