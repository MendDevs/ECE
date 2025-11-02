import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
#################### Page configuration #########
st.set_page_config(layout="wide")
st.title("TP 3 - Classification supervisé")
st.markdown("""----""")
###################### partie 1 #################
st.markdown("# Partie 1: ")
st.markdown("## Analyse Préliminaire")

#loading data
titanic_train = pd.read_csv("titanic_train.csv")
st.markdown("### Data Display")
st.dataframe(titanic_train)


st.markdown("### Preliminary Analysis")

#Attributs:
st.markdown("#### General Information")
with st.expander("Click me to display an overview of the data"):
    info_titanic_train = pd.DataFrame({
        "Colonnes" : titanic_train.columns,
        "Type"     : titanic_train.dtypes.astype(str),
        "Valeurs non nulles" : titanic_train.notnull().sum(),
        "Valeurs nulles" : titanic_train.isnull().sum(),
        "Pourcentage Manquant (%)" : round(titanic_train.isnull().mean()*100,2)
    })
    st.dataframe(info_titanic_train)

st.markdown("**Commentaires :**")
st.write("I'll add comment after...Or in report")

#Affichage de Statistic / Descriptif
with st.expander("Affichage de statistiques descriptives"):
    st.dataframe(titanic_train.describe())

# Visualisation des distributions
st.markdown("### Visualization of distribution")
with st.expander("Click  Me to see visuals"):
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(titanic_train['Age'].dropna(),kde=True,ax=axes[0],color="skyblue")
    axes[0].set_title("Distribution de l'age")

    sns.countplot(x='Pclass', data=titanic_train, palette="viridis", ax=axes[1])
    axes[1].set_title("Repartition selon la classe des passagers")
    st.pyplot(fig)

# Analyse de la survie
st.markdown("### Analyse du taux de survie")
with st.expander("Afficher le nombre de survivants et de morts: "):
    survival_count = titanic_train['Survived'].value_counts()
    survival_count_percent= titanic_train['Survived'].value_counts(normalize=True)*100

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("###### Nombre Total: ")
        st.dataframe(pd.DataFrame(
            {
                "Statut": ["Mort(0)", "Survivant (1)"],
                "Nombre": survival_count.values
            }
        ))
    with col2:
        st.markdown("###### Pourcentage : ")
        st.dataframe(pd.DataFrame({
            "Statut" : ["Mort(0)","Survivant(1)"],
            "Pourcentage (%)" : round(survival_count_percent, 2)
        }))
    
    fig,ax = plt.subplots()
    sns.countplot(x='Survived', data=titanic_train, palette='pastel')
    ax.set_xticklabels(['Mort(0)','Survivant (1)'])
    ax.set_title("Repartition des survivants vs morts")
    st.pyplot(fig)

