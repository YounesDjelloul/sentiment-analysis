import streamlit as st
import joblib

MODEL_PATH = "./models/LogisticRegressionModel.pkl"
VECTORIZER_PATH = "./models/LogisticRegressionModel.pkl"
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

tweet_text = st.text_area("Enter the tweet text:", placeholder="Type your tweet here...")

if st.button("Analyze Sentiment"):
    if tweet_text.strip():
        tweet_vectorized = vectorizer.transform([tweet_text])

        prediction = model.predict(tweet_vectorized)[0]

        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter a valid tweet text.")
