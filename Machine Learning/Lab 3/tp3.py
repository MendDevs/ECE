import streamlit as st 
import pandas as pd 
import numpy as np

#################### Page configuration #########
st.set_page_config(layout="wide")
st.title("TP 3 - Classification supervisé")
st.markdown("""----""")
###################### partie 1 #################

titanic_train = pd.read_csv("titanic_train.csv")
