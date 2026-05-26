import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("Loading dataset...")

fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Label: 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

print(f"Total articles: {len(df)}")
print(f"Fake: {len(fake_df)} | Real: {len(true_df)}")

# ─────────────────────────────────────────────
# 2. PREPROCESS TEXT
# ─────────────────────────────────────────────
print("\nPreprocessing text...")

# Combine title + text for richer context
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# Basic cleaning — remove extra whitespace
df["content"] = df["content"].str.lower().str.strip()
df["content"] = df["content"].str.replace(r"\s+", " ", regex=True)

# Drop rows with empty content
df = df[df["content"].str.len() > 10].reset_index(drop=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
print("\nVectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,       # top 50k most important words
    ngram_range=(1, 2),       # unigrams + bigrams
    stop_words="english",     # remove common English stopwords
    min_df=2,                 # ignore very rare terms
    sublinear_tf=True         # apply log normalization
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL — LOGISTIC REGRESSION
# ─────────────────────────────────────────────
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=5,              # regularization strength
    solver="lbfgs",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# ─────────────────────────────────────────────
# 6. EVALUATE MODEL
# ─────────────────────────────────────────────
print("\n── EVALUATION RESULTS ──────────────────────")

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Fake  (TN): {cm[0][0]}")
print(f"  False Real (FP): {cm[0][1]}")
print(f"  False Fake (FN): {cm[1][0]}")
print(f"  True Real  (TP): {cm[1][1]}")

# ─────────────────────────────────────────────
# 7. SAVE MODEL + VECTORIZER
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save accuracy for the app to display
with open("model/accuracy.txt", "w") as f:
    f.write(f"{accuracy * 100:.2f}")

print(f"\nModel saved to model/model.pkl")
print(f"Vectorizer saved to model/vectorizer.pkl")
print(f"\nDone! Model accuracy: {accuracy * 100:.2f}%")
