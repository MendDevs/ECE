import streamlit as st
import pandas as pd


data ={
    "Species" : ["Bat A", "Bat B", "Bat C"],
    "BOW"     : [120, 150, 100],
    "BRW"     : [2.5, 3.0, 1.8]
}

df = pd.DataFrame(data)
st.dataframe(df)
st.table(df) 
st.dataframe(df.style.highlight_max(axis=0))
