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


![P1220791](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/1680cb64-f05b-41c6-b5d2-b99ba308a921)


![P1220792](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/bd033999-0419-435f-9062-ee1a7a3fac53)


![P1220793](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/83f4655f-26d8-429e-9d0f-c235cbaf0e9b)



La station météo fonctionne en autonomie grâce à un panneau solaire qui alimente une batterie de 46 A/h par le biai d'un régulateur de charge. Pour fonctionner en autonomie la station ne s'allume que dix minutes par heure, le temps de collecter, enregistrer, prédire et transmettre.





31 octobre 2023: j'ai ajouté des scripts suite à la "récolte" d'un an de météo locale; le but cette fois n'est plus d'utiliser les données de MeteoNet et Eco2mix, mais juste les données captées depuis un an par la station pour avoir un modele prédictif sur la meteo locale.
Cette fois, comme on va avoir moins d'échantillons que le jeu MeteoNet (et un pc avec plus de puissance de calcul), on va programmer un seul modele de deep learning RNN (LSTM) pour prédire en même temps toutes nos variables (direction du vent, force, humidité, température et luminosité) et aussi apte à utiliser les phots du ciel comme variable d'entrée.

Etape 1: reformater les images de la raspberry pi

avec le script "reformateur_images.py", on va crée un dossier avec dedans nos images reformatées en 64*64 et en noir et blanc. Dans ce dossier on aura également toutes nos images regroupées sous forme de tenseur; comme elles ne seront pas rangées dans l ordre chronologique, un index sera également fournit dans le dossier. Cette étape permet de réduire la taille des images pour ne pas surcharger le travail de l'encodeur pour extraire les caractéristiques des images.

Etape 2: encoder les images.

Avec le tenseur et son index("auto_encodeur_convolutif_photos_ciel.py"), on va encoder les images via un encodeur convolutif, afin d'extraire un vecteur contenant 960 dimensions de caractéristiques à la place de chaque image. Cela va permettre ensuite de soumettre les images, sous cette forme, à un LSTM qui va traquer l'évolution de ces dimensions dans le temps.

Etape 3: agencer les images pour le LSTM, puis lancer l'entrainement.

Pour le moment je n'ai pas fait une recherche de paramètres très étendue (GridSearchCV) à cause du manque de temps et comme toujours de puissance de calcul, mais avec le modele dans ce script ("traitement_data_meteo_locale.py") on arrive à ces courbes suite à un entrainement d'une quarantaine d'époques:

![modele_rnn_global](https://github.com/HarryTutle/station_meteo_intelligente/assets/82940602/8fac5224-c087-413a-a19e-17c848677ee6)

les courbes en bleu donnent la perte lors de l'entrainement; en rouge la mae toujours pour l'entrainement. En orange la perte pour la validation et en vert la mae pour la validation.

Après il y a très certainement d'autres agencements à tester, plus efficaces pour l'encodage des images, l'architecture du LSTM...A tester !!! Du coup je met en plus sur le git mes données locales(data_total.csv, les fichier index_tensuer et les fichiers compressés tenseur pour les photos du ciel). Amusez-vous bien!


27 fevrier 2024: la raspberry a laché, du coup je profite du petit dépannage pour expliquer le montage de la station (j'aurai pu utiliser fritzing mais ça aurait été illisible il y a pas mal de choses à intégrer):

1) La raspberry pi 3 b+

   on va avoir deux prises usb mobilisées, une pour enregistrer sur la clé usb et l'autre pour communiquer avec l'arduino. Concernant l'émetteur radio son VCC sera branché sur la pin 2 (5v), son GND sur la pin 6 (gnd) et la borne data sera sur la pin 11. Pour l'ecran oled qui affichera la prévision, son VCC sera sur la pin 1 (3.3 volts), son GND sur la pin 14 (gnd), son SDA sur la pin 3 et son SCL sur la pin 5.

2) La arduino uno

   le plus important!!! Pour éviter de faire un reset à chaque ouverture de la communication série entre la arduino et la raspberry pi, il faut brancher un condensateur de 10 microfarads sur la pin reset de l'arduino et le GND. Pour monter l'anémo, girouette et pluviometre j'ai utilisé une platine grove de lextronic où on peut tout connecter dessus. Cette platine se branche sur tout les pins du coté analogique. Le relai qui allume la raspberry est branché sur la pin 4, l'horloge DS3231 a son pin SQW branché sur la pin 2 de l'arduino (gère le réveil).Pour le BME 280 et l'horloge leurs SDA et SCL de liasion I2C sont reliés aux pins SDA et SCL de l'arduino. Le pin A3 accueille le phototransistor, le pin A0 le pluviomètre, le pin A1 la girouette et la pin A2 l'anémo. Pour éviter les bugs il est préférable de relier tout ces composants à une masse commune.

3) autre

   La raspberry pi est branchée sur la batterie par le biai d'un transfo qui lui fournit une tension de 5V et 5A.
   Ce branchement est activé par le relai de l'arduino. La arduion est également branchée à la batterie via un autre transfo pour lui fournir 9V (54W).



Avant de reconditionner la station j'ai viré la caméra pour simplifier l'ensemble; précédemment je trouvai que l'info apportée par les photos du ciel n'apportait pas une grande pluvalue, et amenait en revanche bien plus de complexité superflue. Donc j'ai fait un modèle LSTM pour prédire la direction/force/humidité/température/luminosité juste avec mes données locales qui se sont arretées à décembre (soit un an et demi environ). J'ai mis en telechargement mes données meteo locales, ce nouveau modèle, et un script python pour permettre la création de ce modèle.




