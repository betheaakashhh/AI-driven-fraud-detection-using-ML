# model_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset from path
def load_data(path):
    return pd.read_csv(path)

# Preprocess and balance the dataset
def preprocess_data(data):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=492)  # Undersampling

    balanced_data = pd.concat([legit_sample, fraud], axis=0)
    X = balanced_data.drop(columns='Class', axis=1)
    Y = balanced_data['Class']
    return X, Y, balanced_data

# Split and scale features
def split_and_scale(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler

# Train multiple models
def train_models(X_train, Y_train):
    log_model = LogisticRegression()
    tree_model = DecisionTreeClassifier()
    iso_model = IsolationForest(contamination=0.01)

    log_model.fit(X_train, Y_train)
    tree_model.fit(X_train, Y_train)
    iso_model.fit(X_train)

    return log_model, tree_model, iso_model

# Evaluate selected model
def evaluate_model(model, X_test, Y_test, model_type="supervised"):
    if model_type == "unsupervised":
        preds = model.predict(X_test)
        preds = [0 if x == 1 else 1 for x in preds]
    else:
        preds = model.predict(X_test)
    
    acc = accuracy_score(Y_test, preds)
    conf_matrix = confusion_matrix(Y_test, preds)
    report = classification_report(Y_test, preds, output_dict=True)
    return acc, conf_matrix, report
