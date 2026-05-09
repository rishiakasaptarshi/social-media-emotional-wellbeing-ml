# ============================================================
# PROJECT:
# Social Media Usage and Emotional Well-Being
#
# TYPE:
# Machine Learning + Deep Learning + Transfer Learning
#
# MODELS USED:
# 1. Random Forest
# 2. Deep Learning Neural Network
# 3. HuggingFace DistilBERT
#
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from transformers import pipeline

import joblib

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\n=================================================")
print("LOADING DATASETS")
print("=================================================\n")

# Handles corrupted CSV rows automatically

train_df = pd.read_csv(
    "train.csv",
    engine="python",
    on_bad_lines="skip"
)

val_df = pd.read_csv(
    "val.csv",
    engine="python",
    on_bad_lines="skip"
)

test_df = pd.read_csv(
    "test.csv",
    engine="python",
    on_bad_lines="skip"
)

print("Datasets Loaded Successfully!\n")

print("Train Shape :", train_df.shape)
print("Validation Shape :", val_df.shape)
print("Test Shape :", test_df.shape)

# ============================================================
# COMBINE TRAIN + VALIDATION
# ============================================================

df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

print("\nCombined Dataset Shape :", df.shape)

# ============================================================
# REMOVE NULL VALUES
# ============================================================

df.dropna(inplace=True)
test_df.dropna(inplace=True)

# ============================================================
# DISPLAY SAMPLE DATA
# ============================================================

print("\n=================================================")
print("DATA SAMPLE")
print("=================================================\n")

print(df.head())

# ============================================================
# TARGET COLUMN
# ============================================================

target_column = df.columns[-1]

print(f"\nTarget Column : {target_column}")

# ============================================================
# LABEL ENCODING
# ============================================================

print("\n=================================================")
print("LABEL ENCODING")
print("=================================================\n")

label_encoders = {}

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        combined_values = pd.concat([
            df[column].astype(str),
            test_df[column].astype(str)
        ])

        le.fit(combined_values)

        df[column] = le.transform(
            df[column].astype(str)
        )

        test_df[column] = le.transform(
            test_df[column].astype(str)
        )

        label_encoders[column] = le

print("Encoding Completed!")

# ============================================================
# FEATURES & TARGET
# ============================================================

X_train = df.drop(target_column, axis=1)
y_train = df[target_column]

X_test = test_df.drop(target_column, axis=1)
y_test = test_df[target_column]

# ============================================================
# FEATURE SCALING
# ============================================================

print("\n=================================================")
print("FEATURE SCALING")
print("=================================================\n")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling Completed!")

# ============================================================
# ============================================================
# MODEL 1 : RANDOM FOREST
# ============================================================
# ============================================================

print("\n=================================================")
print("MODEL 1 : RANDOM FOREST")
print("=================================================\n")

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_predictions = rf_model.predict(
    X_test_scaled
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(f"Random Forest Accuracy : {rf_accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    rf_predictions
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/random_forest_confusion_matrix.png"
)

plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n=================================================")
print("FEATURE IMPORTANCE")
print("=================================================\n")

print(feature_importance)

# ============================================================
# SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

print("\nRandom Forest Model Saved!")

# ============================================================
# ============================================================
# MODEL 2 : DEEP LEARNING
# ============================================================
# ============================================================

print("\n=================================================")
print("MODEL 2 : DEEP LEARNING")
print("=================================================\n")

num_classes = len(np.unique(y_train))

y_train_dl = to_categorical(
    y_train,
    num_classes
)

y_test_dl = to_categorical(
    y_test,
    num_classes
)

# ============================================================
# BUILD MODEL
# ============================================================

dl_model = Sequential()

dl_model.add(
    Dense(
        128,
        activation='relu',
        input_shape=(X_train_scaled.shape[1],)
    )
)

dl_model.add(
    Dropout(0.3)
)

dl_model.add(
    Dense(
        64,
        activation='relu'
    )
)

dl_model.add(
    Dropout(0.3)
)

dl_model.add(
    Dense(
        32,
        activation='relu'
    )
)

dl_model.add(
    Dense(
        num_classes,
        activation='softmax'
    )
)

# ============================================================
# COMPILE MODEL
# ============================================================

dl_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# TRAIN MODEL
# ============================================================

history = dl_model.fit(
    X_train_scaled,
    y_train_dl,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# ============================================================
# EVALUATE MODEL
# ============================================================

dl_loss, dl_accuracy = dl_model.evaluate(
    X_test_scaled,
    y_test_dl
)

print(f"\nDeep Learning Accuracy : {dl_accuracy * 100:.2f}%")

# ============================================================
# PREDICTIONS
# ============================================================

dl_predictions = dl_model.predict(
    X_test_scaled
)

dl_predictions = np.argmax(
    dl_predictions,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        dl_predictions
    )
)

# ============================================================
# SAVE DL MODEL
# ============================================================

dl_model.save(
    "models/deep_learning_model.h5"
)

print("\nDeep Learning Model Saved!")

# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Deep Learning Accuracy")

plt.legend()

plt.savefig(
    "outputs/deep_learning_accuracy.png"
)

plt.close()

# ============================================================
# ============================================================
# MODEL 3 : HUGGINGFACE TRANSFORMER
# ============================================================
# ============================================================

print("\n=================================================")
print("MODEL 3 : HUGGINGFACE DISTILBERT")
print("=================================================\n")

print("""
Using pretrained DistilBERT model
for sentiment/emotion inference.
""")

# ============================================================
# CREATE TEXT REPRESENTATION
# ============================================================

sample_rows = test_df.head(20)

text_samples = []

for _, row in sample_rows.iterrows():

    row_text = " ".join(
        [str(value) for value in row.values[:-1]]
    )

    text_samples.append(row_text)

# ============================================================
# LOAD TRANSFORMER MODEL
# ============================================================

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ============================================================
# RUN PREDICTIONS
# ============================================================

hf_results = classifier(text_samples)

print("\n=================================================")
print("HUGGINGFACE RESULTS")
print("=================================================\n")

for i, result in enumerate(hf_results):

    print(f"Sample {i+1}")

    print(f"Prediction : {result['label']}")

    print(f"Confidence : {result['score']:.4f}")

    print("-" * 50)

# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Deep Learning Neural Network",
        "HuggingFace DistilBERT"
    ],

    "Type": [
        "Machine Learning",
        "Deep Learning",
        "Transfer Learning"
    ],

    "Accuracy": [
        round(rf_accuracy * 100, 2),
        round(dl_accuracy * 100, 2),
        "Inference Only"
    ]
})

print("\n=================================================")
print("MODEL COMPARISON")
print("=================================================\n")

print(comparison_df)

# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n=================================================")
print("FINAL PROJECT SUMMARY")
print("=================================================\n")

print("""
1. Random Forest
   - Fast
   - High accuracy
   - Best for tabular datasets

2. Deep Learning
   - Learns complex patterns
   - Better scalability
   - Requires more computation

3. HuggingFace DistilBERT
   - Transfer Learning
   - Uses pretrained transformer
   - NLP-based emotional inference

Project Successfully Demonstrates:
✔ Machine Learning
✔ Deep Learning
✔ Transfer Learning
✔ HuggingFace Integration
✔ Model Comparison
""")

print("\n=================================================")
print("PROJECT EXECUTED SUCCESSFULLY")
print("=================================================\n")