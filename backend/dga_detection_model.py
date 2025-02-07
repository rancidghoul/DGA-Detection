import numpy as np
import pandas as pd
import re
import joblib
from math import log2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
def load_dga_dataset():
    try:
        logger.info("Loading dataset...")
        ds = load_dataset("harpomaxx/dga-detection")
        logger.info("Dataset loaded successfully.")
        return ds
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# Feature Extraction
def entropy(domain):
    prob = [domain.count(c) / len(domain) for c in set(domain)]
    return -sum(p * log2(p) for p in prob)

def extract_features(domain):
    return [
        len(domain),
        len(set(domain)),
        sum(c.isdigit() for c in domain),
        sum(1 for c in domain if c in "aeiou") / len(domain),
        entropy(domain),
    ]

# Train and Evaluate Random Forest Model
def train_random_forest(X_train, y_train, X_test, y_test):
    try:
        logger.info("Training Random Forest model...")
        # Train Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        logger.info("Random Forest Accuracy: %.4f", accuracy_score(y_test, y_pred_rf))
        logger.info("Random Forest ROC-AUC Score: %.4f", roc_auc_score(y_test, y_pred_proba_rf))
        logger.info("Random Forest Classification Report:\n%s", classification_report(y_test, y_pred_rf))
        logger.info("Random Forest Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred_rf))

        # Save the model
        joblib.dump(rf_model, "rf_dga_model.pkl")
        logger.info("Random Forest model saved to 'rf_dga_model.pkl'.")
        return rf_model
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        raise

# Train and Evaluate LSTM Model
def train_lstm(X_seq_train, y_train, X_seq_test, y_test, tokenizer):
    try:
        logger.info("Training LSTM model...")
        # Build LSTM Model
        lstm_model = Sequential([
            Embedding(len(tokenizer.word_index) + 1, 16, input_length=20),
            LSTM(32, return_sequences=True),
            LSTM(16),
            Dense(1, activation="sigmoid")
        ])
        lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        lstm_model.fit(X_seq_train, y_train, epochs=10, batch_size=32, validation_data=(X_seq_test, y_test))

        # Evaluate the model
        y_pred_lstm = (lstm_model.predict(X_seq_test) > 0.5).astype(int)
        y_pred_proba_lstm = lstm_model.predict(X_seq_test)

        logger.info("LSTM Accuracy: %.4f", accuracy_score(y_test, y_pred_lstm))
        logger.info("LSTM ROC-AUC Score: %.4f", roc_auc_score(y_test, y_pred_proba_lstm))
        logger.info("LSTM Classification Report:\n%s", classification_report(y_test, y_pred_lstm))
        logger.info("LSTM Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred_lstm))

        # Save the model
        lstm_model.save("lstm_dga_model.h5")
        logger.info("LSTM model saved to 'lstm_dga_model.h5'.")
        return lstm_model
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        raise

# Main function
def main():
    # Load the dataset
    ds = load_dga_dataset()
    domains = ds['train']['domain']
    labels = ds['train']['label']

    # Convert labels to binary (DGA = 1, Legitimate = 0)
    y_binary = np.array([1 if label != "normal.alexa" else 0 for label in labels])

    # Extract features for Random Forest
    X = np.array([extract_features(domain) for domain in domains])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)

    # Tokenize and pad sequences for LSTM
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(domains)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(domains), maxlen=20)

    # Split sequence data into train and test sets
    X_seq_train, X_seq_test, y_train, y_test = train_test_split(X_seq, y_binary, test_size=0.2, random_state=42)

    # Train and evaluate LSTM model
    lstm_model = train_lstm(X_seq_train, y_train, X_seq_test, y_test, tokenizer)

if __name__ == "__main__":
    main()