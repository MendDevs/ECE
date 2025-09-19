import streamlit as st
st.write("Streamlit is working!")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1,2,3], [4,5,6])
st.pyplot(fig)
