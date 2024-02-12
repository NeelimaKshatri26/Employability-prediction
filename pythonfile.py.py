#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string


# In[2]:


with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)


# In[3]:


with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# In[4]:


st.title("Student Employability")
image = Image.open("studentemployability.jpg")
st.image(image, use_column_width=True)


# In[5]:


st.subheader("Rate communication skills")
user_input = st.text_area("")


# In[6]:


st.subheader("Rate Ability to work")
user_input = st.text_area("")


# In[7]:


st.subheader("Rate Self-Confidence")
user_input = st.text_area("")


# In[8]:


if st.button("Predict"):
    if user_input:
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        st.header("Prediction:")
        if prediction == 5:
            st.subheader("Employable")
        elif prediction == 4:
            st.subheader("Employable")
        elif prediction == 1:
            st.subheader("Less Employable")
    else:
        st.subheader("Please enter a text for prediction.")


# In[ ]:




