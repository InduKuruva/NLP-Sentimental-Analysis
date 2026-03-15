#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Load models
models = pickle.load(open("sentiment_models.pkl","rb"))

lr = models["lr"]
svm = models["svm"]
rf = models["rf"]
xgb = models["xgb"]
mnb = models["mnb"]
vectorizer = models["vectorizer"]

analyzer = SentimentIntensityAnalyzer()

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Page config
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("NLP Sentiment Analysis Dashboard")
st.write("Compare Machine Learning Models with VADER Sentiment Analysis")

# Sidebar
st.sidebar.header("Model Selection")

model_option = st.sidebar.selectbox(
"Choose Model",
("Logistic Regression","SVM","Random Forest","XGBoost","Multinomial Naive Bayes")
)

# Text input
text = st.text_area("Enter Text For Sentiment Analysis")

if st.button("Analyze Sentiment"):

    clean = preprocess(text)
    vect = vectorizer.transform([clean]).toarray()

    # Model selection
    if model_option == "Logistic Regression":
        model = lr
    elif model_option == "SVM":
        model = svm
    elif model_option == "Random Forest":
        model = rf
    elif model_option=="XGBoost":
        model = xgb
    else:
        model=mnb

    prediction = model.predict(vect)[0]

    # Get probability safely
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vect)[0]
    else:
        prob = [0.5, 0.5]

    # Ensure only two values (Negative, Positive)
    if len(prob) >= 2:
        negative_prob = prob[0]
        positive_prob = prob[1]
    else:
        negative_prob = 1 - prob[0]
        positive_prob = prob[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Predicted Sentiment", sentiment)

    with col2:
        st.metric("Positive Probability", f"{prob[1]*100:.2f}%")

    with col3:
        st.metric("Negative Probability", f"{prob[0]*100:.2f}%")

    # Probability dataframe
    prob_df = pd.DataFrame({
        "Sentiment": ["Negative","Positive"],
        "Probability": [prob[0], prob[1]]
    })

    st.subheader("Sentiment Probability Distribution")

    col1,col2 = st.columns(2)

    # Bar Chart
    with col1:
        fig, ax = plt.subplots()
        ax.bar(prob_df["Sentiment"], prob_df["Probability"])
        ax.set_title("Model Probability Distribution")
        st.pyplot(fig)

    # Pie Chart
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(prob_df["Probability"], labels=prob_df["Sentiment"], autopct='%1.1f%%')
        ax2.set_title("Sentiment Share")
        st.pyplot(fig2)

    # VADER analysis
    st.subheader("VADER Sentiment Analysis")

    vader = analyzer.polarity_scores(text)

    vader_df = pd.DataFrame({
        "Score":["Positive","Neutral","Negative","Compound"],
        "Value":[vader['pos'],vader['neu'],vader['neg'],vader['compound']]
    })

    st.dataframe(vader_df)

    # VADER chart
    fig3, ax3 = plt.subplots()
    ax3.bar(vader_df["Score"], vader_df["Value"])
    ax3.set_title("VADER Sentiment Distribution")
    st.pyplot(fig3)

    # Final interpretation
    st.subheader("Final Interpretation")

    if vader['compound'] >= 0.05:
        vader_sentiment = "Positive"
    elif vader['compound'] <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"

    st.write(f"ML Model Prediction: **{sentiment}**")
    st.write(f"VADER Prediction: **{vader_sentiment}**")

st.write("---")
st.write("Project: NLP Sentiment Analysis using Multiple ML Models and VADER")


# In[ ]:




