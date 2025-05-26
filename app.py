import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_news(news_text):
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)
    return prediction[0]

st.title("📰 Fake News Detection")

st.write("Paste a news article below and click 'Predict' to check if it is REAL or FAKE.")

news_input = st.text_area("Enter News Article Text Here", height=300)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        result = predict_news(news_input)
        if result == "REAL":
            st.success("✅ This news is predicted to be REAL.")
        else:
            st.error("⚠️ This news is predicted to be FAKE.")
