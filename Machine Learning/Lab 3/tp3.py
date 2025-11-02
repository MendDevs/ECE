import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
#################### Page configuration #########
st.set_page_config(layout="wide")
st.title("TP 3 - Classification supervisé")
###################### partie 1 #################
st.markdown("# Partie 1: ")
st.markdown("## A) Analyse Préliminaire")
st.markdown("""----""")


#loading data
titanic_train = pd.read_csv("titanic_train.csv")
st.markdown("#### Data Display")
st.dataframe(titanic_train)


st.markdown("#### Preliminary Analysis")

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
st.markdown("#### Visualization of distribution")
with st.expander("Click  Me to see visuals"):
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(titanic_train['Age'].dropna(),kde=True,ax=axes[0],color="skyblue")
    axes[0].set_title("Distribution de l'age")

    sns.countplot(x='Pclass', data=titanic_train, palette="viridis", ax=axes[1])
    axes[1].set_title("Repartition selon la classe des passagers")
    st.pyplot(fig)

# Analyse de la survie
st.markdown("#### Analyse du taux de survie")
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

###########################################################
st.markdown("## B) Les femmes et les enfants d'abord! ")
st.markdown("""----""")

# Survival rate by sex
survival_by_sex = titanic_train.groupby("Sex")["Survived"].mean().reset_index()
survival_by_sex["Survived"] = round(survival_by_sex["Survived"]*100,2)

# Add Child column
titanic_train["Child"] = titanic_train["Age"].apply(lambda x: "child" if x<18 else "adult")

# Survival rate by Child/Adult
survival_by_child = titanic_train.groupby("Child")["Survived"].mean().reset_index()
survival_by_child["Survived"] = round(survival_by_child["Survived"]*100,2)

#Affichage
col1, col2 = st.columns(2)

with col1:
    st.dataframe(survival_by_sex.rename(columns={"Survived": "Taux de survie (%)"}))
    fig, ax = plt.subplots()
    sns.barplot(x="Sex", y="Survived", data=survival_by_sex, palette="Set2", ax=ax)
    ax.set_ylabel("Taux de survie (%)")
    ax.set_title("Taux de survie par sexe")
    st.pyplot(fig)

with col2:
    st.dataframe(survival_by_child.rename(columns={"Survived": "Taux de survie (%)"}))
    fig2, ax2 = plt.subplots()
    sns.barplot(x="Child", y="Survived", data=survival_by_child, palette="coolwarm", ax=ax2)
    ax2.set_ylabel("Taux de survie (%)")
    ax2.set_xlabel("Catégorie d'âge")
    ax2.set_title("Survival comparison: Children vs Adults")
    st.pyplot(fig2)

st.markdown("**Analyse du biais et correction**")
st.markdown("""
    Dans la question précédente, on avait supposé que l'âge seul influeçait le taux de survie, en considérant simplement deux catégories: child (moins de 18 ans) et *adult*(18 ans et plus). Cependant, cette hypothèse est biaisée, car elle ne prend pas en compte d'autres variables importantes, comme le sexe ou la classe cosciale ses passagers.

    En effet, les femmes avaient plus de chances d'être secourues que les hommes, indépendamment de leur âge. Ainsi, dire que 'les enfants survivent mieux que les adultes' sans distinguer entre garçons et filles donne une vision incomplète de la réalité.

    Pour corriger ce biais, j'ai combiné les deux facteurs: âge (Child) et sexe (Sex), puis j'ai réévalué les taux de survie selon ces deux dimensions.

    **Correction and re-evaluation :**
""")
survival_by_child_sex = (
    titanic_train.groupby(["Child", "Sex"])["Survived"]
    .mean()
    .reset_index()
)
survival_by_child_sex["Survived"] = round(survival_by_child_sex["Survived"]*100,2)
col1,col2 = st.columns(2)
with col1:
    st.dataframe(survival_by_child_sex.rename(
        columns={"Survived": "Taux de survie (%)"}
    ))


    
    st.write("""Maintenant: Les femmes (qu'elles soient enfants ou adultes, ont un taux de survie nettemet supérieur. 
    Les hommes enfants ne survivent pas nécessairement mieux que les femmes adultes.
    Cela confirme que la survie dépendait à la fois de l'age et du sexe, et non de l'âge seul.
    """)

with col2:
    fig,ax = plt.subplots()
    sns.barplot(
        x="Child",
        y="Survived",
        hue="Sex",
        data=survival_by_child_sex,
        palette="pastel",
        ax=ax
    )
    ax.set_ylabel("Survival Rate (%)")
    ax.set_xlabel("Categorie d'age")
    ax.set_title("Taux de survie selon l'age et le sexe")
    st.pyplot(fig)

