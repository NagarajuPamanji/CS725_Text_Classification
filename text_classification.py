"""
CS 725 Project: Comparative Analysis of Naïve Bayes and Logistic Regression
for Text Classification.

Team Members:
- Pamanji Nagaraju (25m2011)
- Harshit Singh Bhomawat (25m0786)
- Avi Chourasiya (25m2027)
- Sumit Kumar (25m0759)
- Ketan Patil (25m0788)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess_data():
    """
    Loads the 20 Newsgroups dataset and converts text to TF-IDF vectors.
    """
    print("--- Step 1: Loading and Preprocessing Data ---")
    
    # Load dataset (Train/Test split is predefined)
    print("Fetching 20 Newsgroups dataset...")
    train_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
    
    print(f"Training samples: {len(train_data.data)}")
    print(f"Testing samples: {len(test_data.data)}")
    
    # Vectorization (TF-IDF)
    print("Vectorizing text data (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit on train, transform on test
    X_train = vectorizer.fit_transform(train_data.data)
    X_test = vectorizer.transform(test_data.data)
    
    y_train = train_data.target
    y_test = test_data.target
    
    print(f"Vocabulary size: {X_train.shape[1]} words")
    return X_train, X_test, y_train, y_test, test_data.target_names

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains a model and evaluates its accuracy.
    """
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc*100:.2f}%")
    
    return y_pred, acc

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """
    Plots a confusion matrix heatmap.
    """
    print(f"\nGenerating Confusion Matrix for {title}...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    
    # Save the figure
    filename = "confusion_matrix.png"
    plt.savefig(filename)
    print(f"Confusion matrix saved as '{filename}'")
    plt.show()

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    
    # 2. Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    y_pred_nb, acc_nb = train_evaluate_model(nb_model, X_train, y_train, X_test, y_test, "Naïve Bayes")
    print("\nClassification Report (Naïve Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=class_names))
    
    # 3. Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    y_pred_lr, acc_lr = train_evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")
    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, target_names=class_names))
    
    # 4. Comparison & Plotting
    print("\n--- Final Comparison ---")
    print(f"Naïve Bayes Accuracy: {acc_nb*100:.2f}%")
    print(f"Logistic Regression Accuracy: {acc_lr*100:.2f}%")
    
    # Plot confusion matrix for the better model (Logistic Regression)
    plot_confusion_matrix(y_test, y_pred_lr, class_names, "Logistic Regression")

if __name__ == "__main__":
    main()
