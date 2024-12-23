import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import datetime
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\$\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df


def create_run_folder():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"run_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def visualize_data(dataset, folder_name):
    label_counts = dataset['label'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title("Distribution of Labels")
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.savefig(f"{folder_name}/label_distribution.png")
    plt.close()

    dataset['word_count'] = dataset['cleaned_text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset['word_count'], bins=30, kde=True, color='blue')
    plt.title("Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.savefig(f"{folder_name}/word_count_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='word_count', data=dataset, palette='Set2')
    plt.title("Word Count Distribution by Label")
    plt.xlabel("Labels")
    plt.ylabel("Word Count")
    plt.savefig(f"{folder_name}/label_word_count_distribution.png")
    plt.close()


def train_transformer_model(train_file, val_file, model_name="bert-base-uncased"):
    # Load data
    train_df = load_data(train_file)
    val_df = load_data(val_file)

    # Create folder for saving outputs
    folder_name = create_run_folder()
    visualize_data(train_df, folder_name)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf"
        )

    # Tokenize datasets
    train_encodings = tokenize_function(train_df)
    val_encodings = tokenize_function(val_df)

    train_labels = train_df["label"].to_numpy()
    val_labels = val_df["label"].to_numpy()

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    )).batch(16)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    )).batch(16)

    # Load TensorFlow model
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        batch_size=16
    )

    # Predictions
    predictions = model.predict(val_dataset).logits
    pred_labels = np.argmax(predictions, axis=1)

    # Generate classification report
    report = classification_report(val_labels, pred_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{folder_name}/classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(val_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=train_df['label'].unique(),
        yticklabels=train_df['label'].unique()
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{folder_name}/confusion_matrix.png")
    plt.close()

    return model, history


def train_traditional_model(train_file, val_file):
    train_df = load_data(train_file)
    val_df = load_data(val_file)

    folder_name = create_run_folder()

    visualize_data(train_df, folder_name)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_val = vectorizer.transform(val_df['cleaned_text'])
    y_train = train_df['label']
    y_val = val_df['label']

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{folder_name}/classification_report.csv")

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_df['label'].unique(),
                yticklabels=train_df['label'].unique())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{folder_name}/confusion_matrix.png")
    plt.close()

    accuracy = accuracy_score(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Accuracy"], y=[accuracy], palette='Blues')
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.savefig(f"{folder_name}/accuracy.png")
    plt.close()

    return model, vectorizer


def test(train_file):
    train_df = load_data(train_file)

    folder_name = create_run_folder()

    visualize_data(train_df, folder_name)


if __name__ == "__main__":
    train_file = "dataset/sent_train.csv"
    val_file = "dataset/sent_valid.csv"

    print("Training Transformer Model...")
    transformer_model, transformer_trainer = train_transformer_model(train_file, val_file)

    print("Training Traditional Model...")
    traditional_model, tfidf_vectorizer = train_traditional_model(train_file, val_file)
