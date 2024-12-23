nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                   "N": wordnet.NOUN,
                   "V": wordnet.VERB,
                   "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def clean_text(self, text):
        # Basic cleaning
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\$\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) 
                 for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

def load_data(file_path):
    df = pd.read_csv(file_path)
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    return df

def create_features(text_data, max_features=5000, ngram_range=(1, 2)):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95
    )
    tfidf_features = tfidf.fit_transform(text_data)
    
    # Additional features
    text_length = np.array([len(text.split()) for text in text_data]).reshape(-1, 1)
    
    # Combine features
    return tfidf, tfidf_features, text_length

def train_models(train_file, val_file, folder_name):
    # Load and preprocess data
    train_df = load_data(train_file)
    val_df = load_data(val_file)
    
    # Create features
    tfidf, X_train_tfidf, train_length = create_features(train_df['cleaned_text'])
    X_val_tfidf = tfidf.transform(val_df['cleaned_text'])
    val_length = np.array([len(text.split()) for text in val_df['cleaned_text']]).reshape(-1, 1)
    
    # Scale length features
    scaler = StandardScaler()
    train_length_scaled = scaler.fit_transform(train_length)
    val_length_scaled = scaler.transform(val_length)
    
    # Combine features
    X_train = np.hstack([X_train_tfidf.toarray(), train_length_scaled])
    X_val = np.hstack([X_val_tfidf.toarray(), val_length_scaled])
    
    y_train = train_df['label']
    y_val = val_df['label']
    
    # Define models to try
    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'linear_svc': LinearSVC(random_state=42, dual=False)
    }
    
    best_model = None
    best_accuracy = 0
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'report': classification_report(y_val, y_pred, output_dict=True),
            'predictions': y_pred
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        
        # Save individual model results
        save_model_results(name, y_val, y_pred, folder_name)
    
    # Compare models visually
    plot_model_comparison(results, folder_name)
    
    return best_model, tfidf, results

def save_model_results(model_name, y_true, y_pred, folder_name):
    # Save classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{folder_name}/{model_name}_classification_report.csv")
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{folder_name}/{model_name}_confusion_matrix.png")
    plt.close()

def plot_model_comparison(results, folder_name):
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = [results[model]['accuracy'] for model in results]
    sns.barplot(x=list(results.keys()), y=accuracies, palette='viridis')
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(f"{folder_name}/model_comparison.png")
    plt.close()


if __name__ == "__main__":
    train_file = "dataset/sent_train.csv"
    val_file = "dataset/sent_valid.csv"

    print("Training Traditional Model...")
    best_model, vectorizer, results = train_models(train_file, val_file)