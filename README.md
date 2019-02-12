# Hotword-Detection
Project for Voice Recognition course.

University Paris-Saclay, Master AIC, 2019


## Features et buildset

features.py permet de créer un fichier csv contenant les coefficients mfcc et mfcc-delta (derivatives) de 25 000 enregistrements. Les données sont aussi étiquetées dans la dernière colonne (1 si l'enregistrement correspond à un des trois mots ("one, "two" ou "three") 0 sinon. 

buildset.py réorganise aléatoirement les lignes du fichier csv et créé alors un nouveau fichier


## Script de test

pour Effectuer un test avec micro, executer :

    $ python3 hotword_listener.py 

Cela va charger les modèles entrainés, et lancer un enregistrement d'une seconde (cf indiation dans le terminal) et répondre à l'utilisateur dans le terminal : "hotword detected" ou "not a hotword"

Il est possible en changeant le code de choisir quel réseau est utilisé (1, 2 ou 3 convolutions)



