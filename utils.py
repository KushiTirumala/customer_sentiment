import joblib

def save_model(model, vectorizer, model_file, vectorizer_file):
    """Save trained model and TF-IDF vectorizer."""
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

def load_model(model_file, vectorizer_file):
    """Load trained model and vectorizer."""
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer
