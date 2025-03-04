# Q-Learning Trading Agent

Ce projet est une démonstration simple d'un agent de trading utilisant l'apprentissage par renforcement (Q-learning) pour apprendre une stratégie de trading.  
L'agent est entraîné sur des données historiques de la Bank of America et testé sur des données de General Electric.

## Présentation du Projet

L'objectif est de maximiser la valeur finale d'un portefeuille en simulant des opérations d'achat et de vente d'actions sur une période donnée.  
Le programme lit deux fichiers CSV contenant les prix journaliers :
- **bank_of_america.csv** : utilisé pour entraîner l'agent.
- **ge.csv** : utilisé pour tester la stratégie apprise.

## Comment c'est Fait ?

- **Langage** : Python  
- **Dépendances** :  
  - NumPy  
  - Pandas  
  - Matplotlib  
- **Structure du Code** :  
  Le script `lab2.py` se charge de :
  - Lire et préparer les données.
  - Entraîner l'agent sur les données de la Bank of America.
  - Appliquer la stratégie sur les données de General Electric.
  - Afficher les résultats et générer des graphiques illustrant :
    - Les actions prises (achat, vente, ou maintien).
    - L'évolution du portefeuille.
    - L'évolution des prix avec les signaux d'achat et de vente.

## Résultats

Lors de l'exécution, le programme :
- Affiche la valeur finale du portefeuille obtenue sur les données d'entraînement et de test.
- Génère plusieurs graphiques permettant de visualiser :
  - La série des actions prises par l'agent.
  - La progression de la valeur du portefeuille au fil du temps.
  - Les points d'achat et de vente superposés sur l'évolution des prix.

Ces résultats offrent une vue d'ensemble sur l'efficacité de la stratégie apprise par l'agent et permettent de mieux comprendre son comportement dans un environnement de trading simulé.

## Utilisation

Pour exécuter ce projet :

1. Placez les fichiers `bank_of_america.csv` et `ge.csv` dans le même dossier que le script `lab2.py`.
2. Installez les dépendances si ce n'est pas déjà fait :

   ```bash
   pip install numpy pandas matplotlib
3. Lancez le script :

   ```bash
   pip install numpy pandas matplotlib
