import pandas as pd
import streamlit as st

# --- Build from dict of series ---
d = {
    "one" : pd.Series([1.0,2.0,3.0], index=["a","b","c"]),
    "two" : pd.Series([1.0,2.0,3.0, 4.0], index=["a","b","c","d"])
}

df = pd.DataFrame(d)
#print(st.dataframe(df["two"]))
result=st.dataframe(df)

st.write("========================")
df["one_trunc"] = df["one"][:3]
st.dataframe(df)

# Set indexes
df = pd.DataFrame(d, index=["un","deux","trois","quatre"])
st.dataframe(df)

#Set Columns
df = pd.DataFrame(d,index=["un","deux","trois"], columns=["C1","C2"])
st.dataframe(df)

#from csv
df = pd.read_csv("cleaned_cereal.csv")
st.dataframe(df)
st.write("Text here!")