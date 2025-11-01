import streamlit as st 
import pandas as pd 
import numpy as np
import io
#################### Page configuration #########
st.set_page_config(layout="wide")
st.title("TP 3 - Classification supervisé")
st.markdown("""----""")
###################### partie 1 #################
st.markdown("## Analyse Préliminaire")

#loading data
titanic_train = pd.read_csv("titanic_train.csv")
st.markdown("### Data Display")
st.dataframe(titanic_train)


st.markdown("### Preliminary Analysis")

#Attributs:
st.markdown("#### 1. General Information")
with st.expander("Click me to display an overview of the data"):
    info_titanic_train = pd.DataFrame({
        "Colonnes" : titanic_train.columns,
        "Type"     : titanic_train.dtypes.astype(str),
        "Valeurs non nulles" : titanic_train.notnull().sum(),
        "Valeurs nulles" : titanic_train.isnull().sum(),
        "Pourcentage Manquant (%)" : round(titanic_train.isnull().mean()*100,2)
    })
    st.dataframe(info_titanic_train)