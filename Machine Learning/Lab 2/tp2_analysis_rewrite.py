# ==========================
# Partie A : Chargement et exploration
# ==========================

import streamlit as st
import pandas as pd


st.header("Partie A : Chargement et exploration des données")

# Charger le fichier
try:
    df = pd.read_csv("tabBats.txt", sep="\t")
    st.success("Données chargées avec succès !")
except FileNotFoundError:
    st.error("Le fichier 'tabBats.txt' est introuvable. Vérifiez le chemin.")
    st.stop()

# Aperçu des premières lignes
st.subheader("Aperçu des données")
st.table(df.head())   # petit extrait affiché comme table

# Types de colonnes
st.subheader("Types des colonnes")
st.table(pd.DataFrame(df.dtypes, columns=["Type"]))  # formaté en table

# Statistiques descriptives
st.subheader("Statistiques descriptives")
st.table(df.describe())

# Sélection des variables pour la régression
df_reg = df[["Species", "BOW", "BRW"]].dropna()
st.subheader("Données pertinentes pour la régression")
st.table(df_reg.head())


# ==========================
# Partie B : Première régression linéaire simple
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

sns.set(style="whitegrid")
st.header("Partie B : Première régression linéaire simple")

#1 Nuage de points
st.subheader("")