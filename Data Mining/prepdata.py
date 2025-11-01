import streamlit as st
import matplotlib.pyplot as plt

# Sample data
data = [7, 8, 5, 6, 7, 8, 9, 10, 11, 7, 8, 7, 6, 5, 12, 15, 14, 8, 7, 6]

# Create the figure
plt.figure(figsize=(8, 6))
plt.plot(data, marker='o', linestyle='-', color='b')
plt.title("Simple Data Plot")
plt.xlabel("Index")
plt.ylabel("Value")

# Display in Streamlit
st.pyplot(plt)