import streamlit as st
import joblib
import numpy as np

MODEL_PATH = "./models/best_model.pkl"
VECTORIZER_PATH = "./models/vectorizer.pkl"
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

st.title("Financial Tweets Sentiment Analysis")

st.markdown(
    """
    Enter a tweet below, and our model will predict its sentiment. The sentiments are categorized into the following labels:
    - Positive
    - Negative
    - Neutral
    """
)


def preprocess_input(text, vectorizer):
    tfidf_features = vectorizer.transform([text])
    text_length = np.array([len(text.split())]).reshape(-1, 1)
    features = np.hstack([tfidf_features.toarray(), text_length])

    return features


tweet_text = st.text_area("Enter the tweet text:", placeholder="Type your tweet here...")

if st.button("Analyze Sentiment"):
    if tweet_text.strip():
        processed_features = preprocess_input(tweet_text, vectorizer)

        prediction = model.predict(processed_features)[0]

        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
        predicted_sentiment = sentiment_map[prediction]

        st.success(f"Predicted Sentiment: {predicted_sentiment}")
    else:
        st.warning("Please enter a valid tweet text.")