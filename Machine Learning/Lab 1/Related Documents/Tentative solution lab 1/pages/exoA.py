# pages/1_Exercice_Analyse_Iris.py

import streamlit as st
import pandas as pd
from sklearn import datasets

st.title("A. Manipulation d’un jeu de données")

if 'step_ex1' not in st.session_state:
    st.session_state.step_ex1 = 0

def next_step():
    st.session_state.step_ex1 += 1
def restart():
    st.session_state.step_ex1 = 0
    

col1, col2, col3 = st.columns([1, 2, 4])
with col1:
    st.button("Suivant", on_click=next_step, type="primary")
with col2:
    st.button("Recommencer", on_click=restart)

st.divider()

if st.session_state.step_ex1 >= 1:
    st.header("Étape 1: Chargement des données")
    @st.cache_data
    def load_data():
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target
        iris_df['target_names'] = iris_df['target'].apply(lambda x: iris.target_names[x])
        return iris, iris_df

    iris, iris_df = load_data()
    st.success("Les données Iris ont été chargées avec succès !")

# --- ÉTAPE 2: Affichage des informations de base ---
if st.session_state.step_ex1 >= 2:
    st.header("Étape 2: Affichage des informations de base")
    st.subheader("Données (les 10 premières lignes)")
    st.dataframe(iris_df.head(10))
    st.subheader("Noms des variables (Features)")
    st.write(iris.feature_names)
    st.subheader("Noms des classes (Targets)")
    st.write(iris.target_names)

# ---  3: Nom de la classe pour chaque donnée ---
if st.session_state.step_ex1 >= 3:
    st.header("Étape 3: Nom de la classe pour chaque donnée")
    st.dataframe(iris_df[['target_names'] + iris.feature_names])

# ---  4: Statistiques descriptives ---
if st.session_state.step_ex1 >= 4:
    st.header("Étape 4: Statistiques descriptives par variable")
    stats_df = iris_df.describe().loc[['mean', 'std', 'min', 'max']]
    st.dataframe(stats_df)

# ---  5: Dimensions du jeu de données ---
if st.session_state.step_ex1 >= 5:
    st.header("Étape 5: Dimensions des données (`shape` et `size`)")
    nb_donnees, nb_variables = iris.data.shape
    st.metric(label="Nombre de données", value=nb_donnees)
    st.metric(label="Nombre de variables", value=nb_variables)
    nb_classes = iris.target_names.size
    st.metric(label="Nombre de classes", value=nb_classes)

# --- Message de fin ---
if st.session_state.step_ex1 >= 6:
    st.success("Fin de la solution de l'exercice A. Vous pouvez passer à l'exercice B.")