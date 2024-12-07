# Projet de filtrage anti-spam

## Description

Ce projet de **filtrage** anti-spam est basé sur une application qui utilise le machine learning pour classifier les mails comme spam ou ham. Le déploiement a été effectué avec Streamlit.


### Procédure

- **Etape 1**: Analyse exploratoire des données.
- **Etape 2**: Prétraitement et Transformation du texte avec la bibliothèque NLTK.
- **Etape 3**: Nuage de mots pour les messages SPAM et HAM.
- **Etape 4**: Vectorisation du texte avec TF-IDF.
- **Etape 5**: Construction des modèles.
- **Etape 6**: Evaluation d'un potentiel risque d'over/under fitting.

## Packages utilisés

- **Streamlit** (Framework Python): Pour le server backend.
- **Streamlit** (Frontend): Pour l'interface utilisateur.
- **Machine Learning Packages** (Pandas, Numpy, Nlkt, Spacy, WorlCloud, Scikit-Learn): Utiliser pour le prétrtaitement et la construction du modèle.

## Fonctionnement de l'application

Pour utiliser l'application, veuillez suivre la procédure suivante :

1. **Clonez ce référentiel** : Obtenez le code source du projet en clonant ce référentiel sur votre machine locale.

2. **Installez les packages Python nécessaires** :
   Vous devrez installer quelques packages Python à l'aide de pip. Ouvrez votre terminal et exécutez les commandes suivantes :

   ```bash
   pip install streamlit
   pip install nltk
   ```
3. **Exécutez l'application Streamlit**:
   - Lancez l'application Streamlit en exécutant la commande suivante dans votre terminal :

   ```bash
   python app.py
   ```

   - Ouvrez votre navigateur pour accéder à l'application sur localhost :

   ```bash
      http://192.168.1.104:8501
   ```


