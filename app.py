import os                                                                       
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import show
from streamlit_option_menu import option_menu
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()

import logging

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)

ps = PorterStemmer()

# Charger le modèle pré-entraîné et le vectoriseur
try:
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Erreur lors du chargement du modèle ou du vectoriseur. Veuillez vous assurer que les fichiers 'spam.pkl' et 'vectorizer.pkl' sont présents.")
    st.stop()

def transform_text(text):
    try:
        logging.debug("Début de la transformation du texte.")
        
        # Initialiser le traducteur
        translator = Translator()
        
        # Détection de la langue
        detected_lang = translator.detect(text).lang
        logging.debug(f"Langue détectée : {detected_lang}")
        
        # Traduction en anglais si la langue détectée n'est pas l'anglais
        if detected_lang != 'en':
            text = translator.translate(text, src=detected_lang, dest='en').text
            logging.debug(f"Texte traduit : {text}")
        
        # Conversion du texte en minuscules
        text = text.lower()
        logging.debug(f"Texte en minuscules : {text}")
        
        # Tokenisation du texte
        text = nltk.word_tokenize(text)
        logging.debug(f"Texte tokenisé : {text}")
        
        # Suppression des mots non alphanumériques
        text = [word for word in text if word.isalnum()]
        logging.debug(f"Texte après suppression des mots non alphanumériques : {text}")
        
        # Suppression des stopwords et de la ponctuation
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words and word not in string.punctuation]
        logging.debug(f"Texte après suppression des stopwords et de la ponctuation : {text}")
        
        # Application du stemming
        text = [ps.stem(word) for word in text]
        logging.debug(f"Texte après stemming : {text}")
        
        # Retourner le texte transformé
        transformed_text = " ".join(text)
        logging.debug(f"Texte transformé final : {transformed_text}")
        return transformed_text
    
    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        return ""



st.set_page_config(
    page_title='Spamvanished by Jacquelin & Féridia',
    page_icon="icone.jpg")

# General formating in CSS
page_bg_img = '''
   <style>
   body {
   background-image: url("https://www.xmple.com/wallpaper/black-linear-cyan-gradient-1920x1080-c2-010506-073a47-a-120-f-14.svg");
   background-size: cover;
   color: #fff;
   }
   
   h1 {
   	color:#c4d8d6;
   }
   
   h2 {
   color : #5a6794;
   }
   
   label {
   color: #fff;
   }
   
   .stButton>button {
   color: #000000;
   background-color: #f6be65;
   font-size: large;
   }
   
   .stTextArea>label {
   color: #fff;
   font-size: medium;
   }
   
   .stTextArea>div{
   background-color: #ddddda;
   }
   
   .stTextInput>label {
   color: #fff;
   font-size: medium;
   }
   
   .stTextInput>div>div{
   background-color: #ddddda;
   }
   
   .stSelectbox>label{
   color: #fff;
   font-size: medium;
   }
   
   .stSelectbox>div>div{
   background-color: #ddddda;
   }
   
   .btn-outline-secondary{
   	background-color: #f6be65;
   }
   
   .btn-outline-secondary>span{
   	color: #000000;
   }
   
   .stAlert{
   background-color: #b0cac7;
   }
   </style>
   '''
st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    # Sidebar menu using option_menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu de navigation",  # Sidebar title
            options=["Home", "Description du Dataset", "Procédure d'Apprentissage Automatique", 
                     "Résultats", "Classifier son mail", "Équipe de Développement"],
            icons=["house", "list", "gear", "bar-chart", "envelope", "people"],
            menu_icon="cast",  # Icon for the sidebar
            default_index=0,  # Default selected menu
        )

    # Navigation based on selection
    if selected == "Home":
        page_home()
    elif selected == "Description du Dataset":
        page_dataset()
    elif selected == "Procédure d'Apprentissage Automatique":
        page_procedure()
    elif selected == "Résultats":
        page_results()
    elif selected == "Classifier son mail":
        page_classify()
    elif selected == "Équipe de Développement":
        page_team()


#################
## Page 'Home' ##
#################

import streamlit as st

def page_home(): 
    # Titre principal avec styles personnalisés
    st.markdown("""
        <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .sub-title {
            font-size: 20px;
            color: #34495e;
            text-align: center;
            margin-bottom: 20px;
        }
        .example-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .highlight {
            font-weight: bold;
            color: #e74c3c;
        }
        </style>
        <div class="main-title">Spamvanished, le bouclier ultime pour le filtrage anti-spam</div>
        <div class="sub-title">La seule solution pour sécuriser vos e-mails en temps réel !</div>
    """, unsafe_allow_html=True)

    # Image illustrative
    st.image("hacker.jpeg", caption="Sécurisez vos communications numériques", use_column_width=True)

    # Avantages de la détection de spam
    st.markdown("""
        ## 🚀 Pourquoi utiliser un détecteur de SPAM ?
        - ✅ Évitez les arnaques en ligne.
        - ✅ Protégez vos informations personnelles.
        - ✅ Améliorez votre productivité en filtrant les e-mails indésirables.
    """)

    # Objectifs du projet
    st.subheader("🎯 Objectifs du Projet")
    st.write("""
    Ce projet vise à créer une solution de **filtrage automatique des spams** à l'aide de modèles de **Machine Learning**. 
    Le but est de prédire si un e-mail reçu est un **spam** (indésirable) ou un **ham** (non-spam), afin d'améliorer la gestion des boîtes de réception et de protéger les utilisateurs contre les e-mails malveillants.
    """)

    # Importance du projet
    st.subheader("🧠 Pourquoi ce projet est-il important ?")
    st.write("""
    Avec l'augmentation du nombre de mails reçus chaque jour, il devient difficile de gérer efficacement les boîtes de réception. Les **spams** représentent une part importante des mails reçus et peuvent être source de frustration. L'automatisation de leur filtrage permet non seulement de gagner du temps, mais aussi d'améliorer la sécurité en évitant les mails malveillants.
    """)

    # Domaines d'application concrets
    st.subheader("🌍 Domaines d'application concrets")
    st.write("""
    Ce modèle de classification de mails a plusieurs applications concrètes, notamment :
    
    1. **Boîtes de réception personnelles** : Améliorer l'expérience utilisateur en filtrant automatiquement les spams et en permettant une gestion plus fluide des emails importants.
    
    2. **Entreprises et organisations** : Permettre aux entreprises de réduire les risques liés aux spams malveillants, améliorer la productivité des employés et sécuriser les informations sensibles.
    
    3. **Services financiers** : Automatiser le filtrage des mails pour détecter les tentatives de phishing et les fraudes liées aux courriels.
    
    4. **Systèmes de gestion d'emails à grande échelle** : Optimiser les services de messagerie pour les fournisseurs de services de mails (Gmail, Outlook, etc.), offrant ainsi une meilleure expérience aux utilisateurs finaux.
    
    5. **Filtrage des contenus indésirables** : L'algorithme peut également être appliqué à d'autres types de contenus numériques, comme les commentaires ou les forums, pour éviter la propagation de spams ou de contenus inappropriés.
    """)




###################################
## Page "Description du dataset" ##
###################################

import pandas as pd
import matplotlib.pyplot as plt

def page_dataset():
    # Titre principal
    st.markdown("""
        <style>
        .main-title {
            font-size: 30px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 22px;
            font-weight: bold;
            color: #34495e;
            margin-top: 20px;
        }
        </style>
        <div class="main-title">Description du Dataset</div>
    """, unsafe_allow_html=True)

    # Introduction
    st.write("""
        Ce dataset contient des e-mails classés en deux catégories : **spam** et **ham** (e-mails légitimes).
        Il est utilisé pour entraîner et tester le modèle de filtrage de SPAM. Voici les détails :
    """)

    # Section 1 : Informations clés
    st.markdown("<div class='section-title'>📊 Informations clés</div>", unsafe_allow_html=True)
    st.write("""
        - **Source :** [Kaggle - Spam Emails Dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)  
        - **Taille :** 5 572 e-mails  
        - **Répartition :** Majorité de ham, minorité de spam  
        - **Caractéristiques :** Texte de l'e-mail, catégorie (spam ou ham)
    """)

    # Chargement du dataset
    dataset_path = "spam.csv"
    try:
        data = pd.read_csv(dataset_path, encoding='latin-1')
        data = data.iloc[:, :2]  # Garder seulement les deux premières colonnes (label et message)
        data.columns = ['Label', 'Message']  # Renommer les colonnes pour plus de clarté

        # Section 2 : Aperçu des données
        st.markdown("<div class='section-title'>📋 Aperçu des Données</div>", unsafe_allow_html=True)
        st.dataframe(data.head(10))

        # Télécharger un échantillon du dataset
        st.download_button(
            label="📥 Télécharger un échantillon",
            data=data.head(100).to_csv(index=False).encode('utf-8'),
            file_name='sample_spam_dataset.csv',
            mime='text/csv'
        )

    # Section 3 : Visualisations
        st.markdown("<div class='section-title'>📊 Visualisations</div>", unsafe_allow_html=True)
        
        # Répartition des catégories (spam/ham)
        label_counts = data['Label'].value_counts()
        labels = label_counts.index
        sizes = label_counts.values
        colors = ['#2ecc71', '#e74c3c']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Cercle parfait
        st.pyplot(fig)
    
    
    except FileNotFoundError:
        st.error(f"Le fichier est introuvable au chemin spécifié : {dataset_path}. Veuillez vérifier le chemin.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement du dataset : {e}")

    # Section 4 : Analyse des Caractéristiques des E-mails
    st.markdown("<div class='section-title'>📊 Analyse des Caractéristiques des E-mails</div>", unsafe_allow_html=True)
    st.write("""
        Cette section présente une analyse des caractéristiques textuelles des e-mails en fonction de leur type (spam ou ham) :
        - Nombre moyen de caractères
        - Nombre moyen de mots
        - Nombre moyen de phrases
        Elle inclut également la matrice de 
    """)

    # Visualisation des caractéristiques (caractères, mots, phrases)
    st.markdown("<div class='photo-section'>", unsafe_allow_html=True)

    # Image du nombre moyen de caractères pour spam et ham
    st.image("caracteres.png", caption="Nombre moyen de caractères", use_column_width=True)

    # Image du nombre moyen de mots pour spam et ham
    st.image("mots.png", caption="Nombre moyen de mots", use_column_width=True)

    # Image du nombre moyen de phrases pour spam et ham
    st.image("phrases.png", caption="Nombre moyen de phrases", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


    # Titre de la section avec style
    st.markdown("""
        <style>
        .section-title {
            font-size: 28px;
            font-weight: bold;
            color: #2980b9;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-description {
            font-size: 18px;
            color: #2c3e50;
            text-align: justify;
            margin-bottom: 20px;
        }
        .stImage {
            margin: 20px 0;
        }
        .photo-section {
            display: flex;
            justify-content: space-evenly;
            flex-wrap: wrap;
        }
        </style>
    """, unsafe_allow_html=True)

    # Titre principal de la section
    st.markdown("<div class='section-title'>☁️ Nuage de Mots pour Spam et Ham</div>", unsafe_allow_html=True)

    # Description de la procédure avec un fond coloré
    st.markdown("""
        <div class="section-description">
        Les nuages de mots ont été générés à partir des e-mails, après un prétraitement du texte. 
        Voici les étapes du processus de transformation du texte :
        
        <ul>
            <li><strong>Conversion en minuscules</strong> : Tous les mots sont convertis en minuscules pour uniformiser les textes.</li>
            <li><strong>Tokenisation</strong> : Le texte est découpé en mots individuels (tokens).</li>
            <li><strong>Suppression des mots non-alphanumériques</strong> : Les caractères comme les émojis, les symboles spéciaux (par exemple : `@`, `$`) sont supprimés.</li>
            <li><strong>Filtrage des mots vides (stopwords) et de la ponctuation</strong> : Les mots courants comme "et", "le", "la", etc., sont éliminés.</li>
            <li><strong>Stemming</strong> : Les mots restants sont réduits à leur racine (par exemple, "manger" devient "mang").</li>
        </ul>

        Ces étapes permettent d'obtenir des représentations plus pertinentes du contenu des e-mails, et les nuages de mots générés reflètent les mots les plus fréquents dans les catégories spam et ham.
        </div>
    """, unsafe_allow_html=True)

    # Section d'affichage des nuages de mots avec une interface agréable
    st.markdown("<div class='photo-section'>", unsafe_allow_html=True)

    # Utilisation de widgets pour interactivité : Choix entre afficher les deux nuages
    col1, col2 = st.columns(2)
    
    with col1:
        # Affichage du nuage de mots pour le Spam
        st.image("nuage_spam.png", caption="Nuage de mots pour Spam", use_column_width=True)

    with col2:
        # Affichage du nuage de mots pour le Ham
        st.image("nuage_ham.png", caption="Nuage de mots pour Ham", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Ajouter un bouton pour télécharger les nuages de mots
    st.download_button(
        label="📥 Télécharger le Nuage de Mots pour spam",
        data=open("nuage_spam.png", "rb").read(),
        file_name="nuage_spam.png",
        mime="image/png"
    )
    st.download_button(
        label="📥 Télécharger le Nuage de Mots pour ham",
        data=open("nuage_ham.png", "rb").read(),
        file_name="nuage_ham.png",
        mime="image/png"
    )

    # Section 5 : Lien vers la source
    st.markdown("<div class='section-title'>🔗 Source des Données</div>", unsafe_allow_html=True)
    st.write("""
        Les données utilisées sont issues de Kaggle. Vous pouvez les télécharger ici :  
        [Kaggle - Spam Emails Dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)
    """)


##################################################
## Page "Procédure d'Apprentissage Automatique" ##
##################################################

import streamlit as st

def page_procedure():
    st.write("## 🔧 Apprentissage automatique 🌟")
    st.write("""
    Voici un aperçu de la procédure suivie pour développer notre modèle de classification de mails (spam/ham) :

    1. **Nettoyage et Préparation des Données** :
       - Vérification des valeurs manquantes : Identifier et traiter les données absentes.
       - Suppression des doublons : S'assurer que chaque observation est unique.
    
    2. **Analyse Exploratoire des Données (EDA)** :
       - Analyse univariée : Étudier les distributions de chaque variable.
       - Analyse bivariée : Identifier les relations entre les variables.
       - Tokenisation : Conversion des textes en tokens pour pouvoir les traiter par les modèles.

    3. **Prétraitement et Transformation du Texte** :
       - Nettoyage des textes : Suppression des caractères spéciaux, conversion en minuscules, etc.
       - Utilisation de la bibliothèque NLTK pour effectuer le prétraitement (lemmatisation, suppression des stopwords).

    4. **Vectorisation avec TF-IDF** :
       - La vectorisation des données textuelles est effectuée à l'aide de la méthode TF-IDF (Term Frequency-Inverse Document Frequency).

    5. **Séparation du Dataset (Train/Test)** :
       - Entraînement : 80% des données sont utilisées pour entraîner le modèle.
       - Test : 20% des données sont utilisées pour évaluer la performance du modèle.

    6. **Construction des Modèles** :

    """)
    
    table_html = """
    <table style="width:100%; border:1px solid #ddd; border-collapse: collapse;">
        <thead>
            <tr>
                <th style="padding: 8px; background-color: #f2f2f2;">Modèle</th>
                <th style="padding: 8px; background-color: #f2f2f2;">Description</th>
                <th style="padding: 8px; background-color: #f2f2f2;">Avantages</th>
                <th style="padding: 8px; background-color: #f2f2f2;">Limites</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 8px;">SVM (Support Vector Machine)</td>
                <td style="padding: 8px;">Modèle supervisé qui cherche à trouver un hyperplan optimal séparant les classes dans un espace de caractéristiques. Utilise des noyaux pour les données non linéaires.</td>
                <td style="padding: 8px;">Efficace dans des espaces de haute dimension. Gère les problèmes non linéaires grâce aux noyaux.</td>
                <td style="padding: 8px;">Lent pour les grands ensembles de données. Nécessite un ajustement des hyperparamètres.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">Naive Bayes</td>
                <td style="padding: 8px;">Modèle probabiliste basé sur le théorème de Bayes et l'indépendance conditionnelle des caractéristiques.</td>
                <td style="padding: 8px;">Rapide, efficace pour des grands ensembles de données et les problèmes de texte.</td>
                <td style="padding: 8px;">Indépendance irréaliste pour certaines tâches.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">GaussianNB()</td>
                <td style="padding: 8px;">Utilise une distribution gaussienne pour les données continues.</td>
                <td style="padding: 8px;">Simple et rapide pour les données continues.</td>
                <td style="padding: 8px;">Moins efficace si les données ne suivent pas une distribution gaussienne.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">MultinomialNB()</td>
                <td style="padding: 8px;">Modèle basé sur la distribution multinomiale, idéal pour la classification de texte (ex : classification de documents).</td>
                <td style="padding: 8px;">Efficace pour la classification de texte.</td>
                <td style="padding: 8px;">Moins performant avec des données non catégorielles.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">BernoulliNB()</td>
                <td style="padding: 8px;">Utilise des données binaires, souvent pour des tâches comme la classification de texte avec présence/absence de caractéristiques.</td>
                <td style="padding: 8px;">Efficace pour des données binaires ou de type présence/absence.</td>
                <td style="padding: 8px;">Moins efficace pour des données continues ou non binaires.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">Régression Logistique</td>
                <td style="padding: 8px;">Modèle linéaire utilisé pour la classification binaire, prédit la probabilité de se retrouver ou non dans une classe.</td>
                <td style="padding: 8px;">Simple, interprétable, efficace pour des tâches de classification binaire.</td>
                <td style="padding: 8px;">Peut sous-performer avec des relations non linéaires complexes.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">Arbre de Décision</td>
                <td style="padding: 8px;">Utilise un arbre binaire pour classer les données en fonction des valeurs des caractéristiques.</td>
                <td style="padding: 8px;">Interprétable, efficace pour des relations non linéaires, pas besoin de normalisation.</td>
                <td style="padding: 8px;">Sensible au bruit et au surajustement avec des arbres trop profonds.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">Forêt Aléatoire</td>
                <td style="padding: 8px;">Plusieurs arbres de décision où chaque arbre est entraîné sur des sous-ensembles de données et de caractéristiques. La prédiction est obtenue par un vote majoritaire.</td>
                <td style="padding: 8px;">Résistant au surajustement, robuste pour des ensembles complexes, gère bien les données manquantes.</td>
                <td style="padding: 8px;">Moins interprétable que les arbres individuels, computationnellement coûteux pour de très grands ensembles.</td>
            </tr>
            <tr>
                <td style="padding: 8px;">K-Nearest Neighbors (K-NN)</td>
                <td style="padding: 8px;">Modèle de classification basé sur la distance, où un exemple est classé par la majorité des voisins les plus proches.</td>
                <td style="padding: 8px;">Simple, facile à comprendre, efficace pour des petites données.</td>
                <td style="padding: 8px;">Lent pour de grands ensembles, dépend fortement du choix de la distance, nécessite beaucoup de mémoire.</td>
            </tr>
        </tbody>
    </table>
    """

    # Afficher le tableau HTML dans l'interface Streamlit
    st.markdown(table_html, unsafe_allow_html=True)

    st.write(""" 

    7. **Évaluation des Modèles** :
       - Considérant le déséquilibre observé au niveau de la target, les métriques F1 Score et AUC ont été utilisées pour évaluer les performances.

    8. **Sélection du Modèle Final** :
       - Le modèle ayant obtenu les meilleures performances a été choisi pour les prédictions.

    9. **Vérification de l'Overfitting et de l'Underfitting** :
       - Observation des courbes d'apprentissage et des scores sur les données de validation pour éviter l'overfitting et l'underfitting.

    10. **Sérialisation du Modèle** :
        - Le meilleur modèle a été sérialisé en fichier .pkl pour être utilisé dans le script de déploiement.
    """)





########################################
########## Page "Résultats" ############
########################################


def page_results():
    st.write("<h3 style='color: #ff6347;'>🌟 Résultats de l'Évaluation du Modèle 🌟</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        Après avoir entraîné et évalué plusieurs modèles, voici les résultats détaillés des performances des différents modèles utilisés pour la classification de mails (spam/ham) :
    """, unsafe_allow_html=True)

    # Résultats des différents modèles sous forme de tableau avec fond blanc
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-top: 20px;">
        <p style="font-size: 18px; color: #2f4f4f;">
        </p>
        <table style="width: 100%; text-align: left; border-collapse: collapse; background-color: #ffffff;">
            <thead>
                <tr style="background-color: #ff6347; color: white;">
                    <th>Modèle</th>
                    <th>F1 Score</th>
                    <th>AUC</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>BernoulliNB</td>
                    <td style="color: #4682b4;">0.888283</td>
                    <td style="color: #4682b4;">0.902725</td>
                </tr>
                <tr>
                    <td>SVC</td>
                    <td style="color: #4682b4;">0.862637</td>
                    <td style="color: #4682b4;">0.886762</td>
                </tr>
                <tr>
                    <td>Random Forest</td>
                    <td style="color: #4682b4;">0.853933</td>
                    <td style="color: #4682b4;">0.875497</td>
                </tr>
                <tr>
                    <td>MultinomialNB</td>
                    <td style="color: #4682b4;">0.835735</td>
                    <td style="color: #4682b4;">0.858911</td>
                </tr>
                <tr>
                    <td>Logistic Regression</td>
                    <td style="color: #4682b4;">0.732143</td>
                    <td style="color: #4682b4;">0.800381</td>
                </tr>
                <tr>
                    <td>Decision Tree</td>
                    <td style="color: #4682b4;">0.648318</td>
                    <td style="color: #4682b4;">0.755339</td>
                </tr>
                <tr>
                    <td>GaussianNB</td>
                    <td style="color: #4682b4;">0.627451</td>
                    <td style="color: #4682b4;">0.841225</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Meilleur modèle et son évaluation
    st.markdown("""
        Le modèle ayant obtenu les meilleures performances globales est le modèle **BernoulliNB (Bernoulli Naive Bayes)**, avec un F1 Score de **0,88** et une AUC de **0,90**.
        Il a été retenu comme modèle final pour prédire si un mail reçu est **spam** ou **ham**. Ce modèle présente la meilleure combinaison de précision et de rappel parmi tous les modèles évalués.
    """, unsafe_allow_html=True)




###################################
## Section "Classifier son mail" ##
###################################

def page_classify():
    
    # Configuration de l'application Streamlit
    st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 24px;
            color: #FF5733;
            margin-bottom: 10px;
        }
        .description {
            font-size: 18px;
            color: #555;
            margin-bottom: 20px;
        }
        .btn-classify {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .btn-classify:hover {
            background-color: #45a049;
        }
        .result-success {
            background-color: #28a745;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .result-error {
            background-color: #dc3545;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # Titre principal
    st.markdown('<div class="main-title">📧 Effectuez le filtrage anti-spam de vos mails </div>', unsafe_allow_html=True)

    # Description
    st.markdown('<div class="description"> Cette application utilise l\'apprentissage automatique pour déterminer si votre e-mail est <b>Spam</b> ou <b>Non Spam (Ham)</b>.</div>', unsafe_allow_html=True)

    # Zone de saisie pour l'utilisateur
    st.subheader("📥 Entrez le texte de l'e-mail")
    user_input = st.text_area("Entrez le texte de l'e-mail ci-dessous pour la classification :", height=150)

    import logging
from googletrans import Translator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import streamlit as st

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)

ps = PorterStemmer()

def transform_text(text):
    try:
        logging.debug("Début de la transformation du texte.")
        
        # Initialiser le traducteur
        translator = Translator()
        
        # Détection de la langue
        detected_lang = translator.detect(text).lang
        logging.debug(f"Langue détectée : {detected_lang}")
        
        # Traduction en anglais si la langue détectée n'est pas l'anglais
        if detected_lang != 'en':
            text = translator.translate(text, src=detected_lang, dest='en').text
            logging.debug(f"Texte traduit : {text}")
        
        # Conversion du texte en minuscules
        text = text.lower()
        logging.debug(f"Texte en minuscules : {text}")
        
        # Tokenisation du texte
        text = nltk.word_tokenize(text)
        logging.debug(f"Texte tokenisé : {text}")
        
        # Suppression des mots non alphanumériques
        text = [word for word in text if word.isalnum()]
        logging.debug(f"Texte après suppression des mots non alphanumériques : {text}")
        
        # Suppression des stopwords et de la ponctuation
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words and word not in string.punctuation]
        logging.debug(f"Texte après suppression des stopwords et de la ponctuation : {text}")
        
        # Application du stemming
        text = [ps.stem(word) for word in text]
        logging.debug(f"Texte après stemming : {text}")
        
        # Retourner le texte transformé
        transformed_text = " ".join(text)
        logging.debug(f"Texte transformé final : {transformed_text}")
        return transformed_text
    
    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        return ""

    # Interface Streamlit
    user_input = st.text_area("Entrez le texte de l'e-mail ci-dessous pour la classification :", height=150)
    
    # Bouton de classification personnalisé
    classify_button = st.button("🔍 Classifier", key="classify_button", help="Cliquez ici pour classifier l'e-mail", use_container_width=True)
    
    if classify_button:
        if user_input.strip():  # Vérifier si l'entrée n'est pas vide
            try:
                data = [user_input]
                transformed_data = [transform_text(text) for text in data]  # Transformer l'entrée à l'aide de la fonction transform_text
                
                if not all(transformed_data):  # Vérifier si la transformation a réussi
                    st.error("Erreur lors de la transformation du texte.")
                else:
                    logging.debug(f"Texte transformé pour vectorisation : {transformed_data}")
                    vec = cv.transform(transformed_data).toarray()  # Transformer l'entrée à l'aide du vectoriseur
                    logging.debug(f"Vecteur transformé : {vec}")
                    result = model.predict(vec)  # Prédire à l'aide du modèle chargé
                    logging.debug(f"Résultat de la classification : {result}")
    
                    # Afficher le résultat avec couleurs personnalisées
                    if result[0] == 0:
                        st.markdown('<div class="result-success">✅ Ce n\'est PAS un e-mail Spam !</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-error">🚨 C\'est un e-mail SPAM !</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la classification : {e}")
        else:
            st.warning("⚠️ Veuillez entrer un texte d'e-mail avant de procéder à la classification.")



######################################
## Section "Equipe de développemnt" ##
######################################


# Définition de la fonction page_team
def page_team():
    # Titre de la section
    st.title("🔍 Équipe de Développement ")

    # Description générale de l'équipe
    st.markdown(
        """
        **Bienvenue dans notre équipe !**  
        Nous sommes une équipe d'**Ingénieurs Statisticiens Économistes (ISE)** passionnés par l'**intelligence artificielle**, les **statistiques** et l'**économie appliquée**.  
        En se basant sur les données, notre objectif est de proposer des solutions innovantes pour résoudre des problèmes spécifiques et améliorer le bien-être des populations.
        """
    )

    # Section pour mettre en valeur les membres de l'équipe
    st.markdown("<h3 style='color: #1f77b4;'>👨‍💻 Membres de l'Équipe 👩‍💻</h3>", unsafe_allow_html=True)

    # Premier membre de l'équipe
    st.markdown(
        """
        **Nom du Développeur 1 : HOUNSOU Jacquelin**  
        - **Profil :** Ingénieur Statisticien Economiste (ISE)   
        - [📚 GitHub](https://github.com/MARIEL-J) | [📧 Email](mailto:hounsoujacquelin@gmail.com)
    """, unsafe_allow_html=True)

    # Deuxième membre de l'équipe
    st.markdown( 
    """
    **Nom du Développeur 2 : AKINDELE Féridia**  
    - **Profil :** Ingénieure Statisticienne Economiste (ISE) 
    - [📚 GitHub](https://github.com/Winrid-lang) | [📧 Email](mailto:feridiaakindele@gmail.com)
    """, unsafe_allow_html=True)

    # Visuels et graphismes : Ajout d'images ou avatars des membres
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #2ecc71;'>Notre Vision :</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        Notre équipe est dédiée à l'application des **méthodes statistiques** et des **algorithmes de machine learning** pour résoudre des problèmes complexes du quotidien.  
        Nous croyons fermement à l'importance de l'**innovation**, de la **collaboration** et de la **transparence** dans notre travail.
        """
    )

    # Citation inspirante
    st.markdown(
        """
        _"Les données ne mentent pas, mais l'interprétation des données fait toute la différence."_
        """
    )

    # Ajout d'un visuel inspirant, une image symbolisant l'innovation, les statistiques ou l'IA
    st.image("ISE.png", caption="L'expertISE au service du développement !", width=300)

    # Bouton de contact
    st.markdown("<hr>", unsafe_allow_html=True)
    st.button("💬 Nous Contacter", on_click=lambda: st.write("Merci pour votre message ! Nous vous répondrons dans les plus brefs délais."))
    
    # Ajout de sections interactives ou graphiques sur l'impact de vos travaux
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        ### 📊 Notre Impact :
        En tant qu'**Ingénieurs Statisticiens Économistes**, nous appliquons nos compétences pour résoudre des défis complexes à travers l'analyse des données et la modélisation.  
        Grâce à notre expertise, nous avons pu créer des modèles **prédictifs** efficaces pour la classification des e-mails, un domaine où l'**apprentissage automatique** peut vraiment faire la différence.
        """
    )

        
def _set_block_container_style(
    max_width: int = 700, max_width_100_percent: bool = False):
    if max_width_100_percent:
        max_width_str = f"max-width: 95%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
</style>
""",
        unsafe_allow_html=True,
    )  

main()
