# station_meteo_intelligente
 les ingrédients pour fabriquer une station meteo IA maison pour prévoir la météo locale.
 
 Actuellement le projet est dans mon jardin et tourne depuis huit mois. La station météo actuelle tourne en enregistrant la météo locale (direction, force, humidité, pluviométrie, température, pression et photo du ciel) toute les heures grâce à une carte arduino uno qui collecte via des composants (horloge externe ds1302, capteur bme280, phototransistor, anémo, girouette) les données pour les envoyer à une raspberry pi 3 B+ chargée d'enregistrer les données sur clé usb, prendre une photo du ciel et calculer la prévision pour les prochaines 24 heures. Enfin, elle envoie via un signal radio la météo actuelle et la prévision à un récepteur qui tourne avec une carte arduino simple.
Mes modèles prédictifs chargés sur la raspberry pi sont entrainés sur un pc fixe suffisamment puissant pour les phases d'entrainement.

A ce stade la station utilise cinq modèles de deep learning en architecture séquentielle, une architecture RNN serait mieux mais mon pc fixe manque de puissance, surtout lorsque j'ajoute du dropout. Il y a un modèle pour prévoir la direction du vent, un autre pour la force, l'humidité, la température et enfin la luminosité. Ces modèles sont entrainés sur les données de MeteoNet (météo France) et Eco2mix (la partie production énergétique des panneaux solaires a servit à évaluer la luminosité). Les données sont traitées et agencées en approche directe.

A l'avenir, les modèles prédictifs vont aussi etre entrainé avec les photos du ciel pour améliorer la prévision ainsi qu'avec les données déjà enregistrées. L'objectif est d'avoir une station météo de plus en plus apte à prévoir la météo locale (les modèles sont de temps en temps réentrainés avec les données locales enregistrées).

Présentation des fichiers:

- Meteo_terre_traitement_explicatif.py: c'est une classe qui permet de modifier les données brutes de MeteoNet et Eco2mix pour avoir un dataset prêt à l'emploi pour le deep learning/machine learning.
- station_meteo_intelligente_5.py: ce programme permet de faire tourner la raspberrypi dans la station météo.
- france_régions_frontières: programme pour obtenir le barycentre des régions Françaises.
- var_soleil_région.csv: fichier csv contenant la luminosité par heure par régions.
- régions_barycentres.csv: fichier indiquant le barycentre de chaque région Française en latitude et longitude.
- arduino_capteurs_corrig_3.ino: programme arduino qui fait tourner la arduino uno de la station chargée de capter les données des capteurs.
- recepteur_radio_data.ino: programme arduino pour le récepteur radio.
- voir un peu sur terre keras total.py: entrainement des réseaux séquentiels, à charger ensuite sur la raspberrypi.
- voir un peu sur terre RNN.py: si vous avez plus de puissance de calcul, entrainement pour des réseaux RNN.
- voir un peu sur terre forest total.py: entrainement avec des forets aléatoires. Prend trop d'espace mémoire pour la raspberrypi.

A venir:

- plans de construction de la station.
- traitement des images via un encodeur convolutif.
- nouveaux modèles prédictifs basés sur les données enregistrées depuis bientôt un an.

Quelques photos:

![P1220783](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/878bcd59-e932-4700-8f15-a7d43d5e1f29)


![P1220784](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/b9c15c6c-06cd-43db-b495-82a9e9d116e0)


![P1220785](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/253b4bc7-24ab-4e3f-9592-3e505c2cc036)


![P1220786](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/081048ca-586f-4caf-82b8-2ccc9d779cec)


![P1220779](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/59774084-d99e-4781-a188-96ecc9751fd5)


![P1220778](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/990126ef-150b-478d-b6e2-43e095a379f0)


![P1220776](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/910453bd-c5cf-478a-92bd-9008040c05c7)








