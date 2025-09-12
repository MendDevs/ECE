import streamlit as st       #  for 
from sklearn import datasets #  pip install scikit-learn
import pandas as pd          #  pip install pandas
import numpy as np           #  pip install numpy

# Titre de l'app
st.title("Exploration du jeu de données Iris")

# 1. Charger les données
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Classe'] = [iris.target_names[i] for i in iris.target]

# 2. Afficher les données
st.subheader("Données Iris (les 5 premières lignes)")
st.dataframe(df.head())

# 3. Noms des variables et classes
st.subheader("Noms des variables (features)")
st.write(iris.feature_names)

st.subheader("Noms des classes (targets)")
st.write(iris.target_names)

# 4. Classe pour chaque donnée
st.subheader("Classe associée à chaque donnée")
st.write(df[['Classe']].head(10))  # Affiche seulement les 10 premières pour la lisibilité

# 5. Statistiques
st.subheader("Statistiques pour chaque variable")
stats = df.describe().transpose()[['mean','std','min','max']]
st.dataframe(stats)

# 6. Taille et dimensions
st.subheader("Taille et dimensions")
st.write(f"Nombre de données : {df.shape[0]}")
st.write(f"Nombre de variables : {df.shape[1]-1}")  # exclut la colonne Classe
st.write(f"Nombre de classes : {len(iris.target_names)}")
