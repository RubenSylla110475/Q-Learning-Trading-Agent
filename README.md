# Q-Learning Trading Agent

Ce dépôt contient une implémentation d'un **agent de trading basé sur Q-learning**.  
L'agent est entraîné sur des données historiques de la **Bank of America (BAC)** et testé sur des données de **General Electric (GE)**.  
L'objectif est d'apprendre une politique (suite d'actions) qui maximise la valeur finale du portefeuille en simulant des opérations d'achat et de vente sur des données journalières.

## Table des matières

- [Description](#description)
- [Dépendances](#dépendances)
- [Utilisation](#utilisation)
- [Structure du code](#structure-du-code)
- [Explication de l'algorithme](#explication-de-lalgorithme)
  - [Fonctions clés](#fonctions-clés)
  - [La signification de Q(s,a)](#la-signification-de-qs-a)
  - [Choix des hyperparamètres](#choix-des-hyperparamètres)
- [Visualisation](#visualisation)
- [Auteurs et Licence](#auteurs-et-licence)

## Description

Ce projet implémente un algorithme de Q-learning pour le trading d'actions.  
L'agent est entraîné sur des données historiques de la **Bank of America (BAC)** et testé sur des données de **General Electric (GE)**.

L'agent peut prendre trois actions :
- **Hold (0)** : ne rien faire.
- **Buy (1)** : acheter 10 actions.
- **Sell (2)** : vendre 10 actions.

La récompense est définie comme la variation quotidienne de la valeur du portefeuille.  
Le but est de maximiser cette valeur finale en apprenant la meilleure séquence d'actions à partir des états observés.

## Dépendances

Pour exécuter ce projet, vous devez avoir installé :
- Python 3.x
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

Vous pouvez installer ces dépendances via pip :

```bash
pip install numpy pandas matplotlib
