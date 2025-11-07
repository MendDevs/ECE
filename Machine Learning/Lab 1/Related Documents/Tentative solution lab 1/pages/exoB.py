# pages/2_Exercice_Suivant.py

import streamlit as st
from sklearn.datasets import fetch_openml

st.title("Partie B : Téléchargement et importation de données")

@st.cache_data 
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data[:1500]
    y = mnist.target[:1500]
    return X, y

X, y = load_data()

st.write("Data shape:", X.shape)
st.write("Labels shape:", y.shape)

anImage = X[4]
anImage_reshaped = anImage.reshape(28, 28)
first_label = y[4]

st.title("Visualisation of the first MNIST image")

normalized_image = anImage_reshaped / 255.0

st.image(
    normalized_image, 
    caption=f"The first image from the MNIST dataset (Label: {first_label})",
    width=300
)