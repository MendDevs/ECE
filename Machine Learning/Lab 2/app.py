import pandas as pd
import streamlit as st

############### Part A: Chargement ex exploration de données ##########
df = pd.read_csv(
                 "tabBats.txt",
                 sep=r"\s+", #seperator = spaces
                 quotechar='"'
                 )

st.dataframe(df)