import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    """Load CSV reviews data."""
    df = pd.read_csv(file_path)
    return df

def clean_text(text):
    """Basic text cleaning: lowercase, remove special chars, numbers."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess_data(df, text_column, target_column):
    """Clean text and split data."""
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    
    X = df[text_column]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
