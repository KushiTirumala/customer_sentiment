from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(X_train, y_train, model_type='NaiveBayes'):
    """Train classifier."""
    if model_type == 'NaiveBayes':
        model = MultinomialNB()
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate classifier."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
