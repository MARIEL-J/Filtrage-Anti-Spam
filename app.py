import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Erreur lors du chargement du modèle ou du vecteur. Assurez-vous que les fichiers 'spam.pkl' et 'vectorizer.pkl' sont présents.")
    st.stop()



# Set up the Streamlit app
st.title("📧 Application de filtrage anti-spam")
st.write(
    """
    Bienvenue dans l'application de classification des emails !  
    Cette application utilise l'apprentissage automatique pour déterminer si un email est **Spam** ou **Non-Spam (Ham)**.  
    """
)

# Input area for the user
st.subheader("Classification")
user_input = st.text_area("Entrez le texte de l'email ci-dessous pour classification:", height=150)

# Classify button
if st.button("Classer"):
    if user_input.strip():  # Check if input is not empty
        # Preprocess and predict
        try:
            data = [user_input]
            vec = cv.transform(data).toarray()  # Transform input using the vectorizer
            result = model.predict(vec)  # Predict using the loaded model
            
            # Display the result
            if result[0] == 0:
                st.success("✅ Ce n'est pas un email Spam !")
            else:
                st.error("🚨 C'est un email SPAM !")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la classification : {e}")
    else:
        st.warning("⚠️ Veuillez entrer le texte de l'email avant de classer.")

# Texte de copyright en bas de la page
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #555;
        }
    </style>
    <div class="footer">
        <p>© Décembre 2024 Réalisé par Jacquelin HOUNSOU & Féridia AKINDELE</p>
    </div>
    """, unsafe_allow_html=True)