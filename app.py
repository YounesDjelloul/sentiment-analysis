import streamlit as st
import joblib
import numpy as np
from textblob import TextBlob
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

MODEL_PATH = "./models/svm.pkl"
VECTORIZER_PATH = "./vectorizers/vectorizers.pkl"
model = joblib.load(MODEL_PATH)
vectorizers = joblib.load(VECTORIZER_PATH)

st.title("Financial Tweets Sentiment Analysis")

st.markdown(
    """
    Enter a tweet below, and our model will predict its sentiment. The sentiments are categorized into the following labels:
    - Positive
    - Negative
    - Neutral
    """
)


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\$\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()

        tokens = word_tokenize(text)

        tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))
                  for token in tokens if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)


def preprocess_input(text, vectorizers):
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)

    tfidf_features1 = vectorizers[0].transform([cleaned_text]).toarray()
    tfidf_features2 = vectorizers[1].transform([cleaned_text]).toarray()
    tfidf_features3 = vectorizers[2].transform([cleaned_text]).toarray()

    sentiment = TextBlob(cleaned_text).sentiment
    sentiment_features = np.array([[
        sentiment.polarity,
        sentiment.subjectivity
    ]])

    financial_features = np.array([[
        cleaned_text.count('$'),
        len(re.findall(r'\d+', cleaned_text)),
        len(re.findall(r'(up|down|rise|fall|gain|loss)', cleaned_text.lower())),
        len(re.findall(r'(bull|bear|bullish|bearish)', cleaned_text.lower()))
    ]])

    features = np.hstack([
        tfidf_features1,
        tfidf_features2,
        tfidf_features3,
        sentiment_features,
        financial_features
    ])

    return features


tweet_text = st.text_area("Enter the tweet text:", placeholder="Type your tweet here...")

if st.button("Analyze Sentiment"):
    if tweet_text.strip():
        processed_features = preprocess_input(tweet_text, vectorizers)
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
