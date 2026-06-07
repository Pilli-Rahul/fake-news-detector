import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────
print("Loading datasets...")

def load_dataset(fake_path, true_path, source_name=""):
    """Load a fake/true CSV pair and return a labeled dataframe."""
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    fake_df["label"] = 0
    true_df["label"] = 1
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    print(f"  [{source_name}] Fake: {len(fake_df)} | Real: {len(true_df)}")
    return combined

datasets = []

# Root dataset
if os.path.exists("dataset/Fake.csv") and os.path.exists("dataset/True.csv"):
    datasets.append(load_dataset("dataset/Fake.csv", "dataset/True.csv", "Root"))
else:
    print("  [Root] Not found, skipping...")

# ISOT dataset
if os.path.exists("dataset/ISOT/Fake.csv") and os.path.exists("dataset/ISOT/True.csv"):
    datasets.append(load_dataset("dataset/ISOT/Fake.csv", "dataset/ISOT/True.csv", "ISOT"))
else:
    print("  [ISOT] Not found, skipping...")

# Future datasets — uncomment when ready
# if os.path.exists("dataset/WELFake/Fake.csv") and os.path.exists("dataset/WELFake/True.csv"):
#     datasets.append(load_dataset("dataset/WELFake/Fake.csv", "dataset/WELFake/True.csv", "WELFake"))

if not datasets:
    raise FileNotFoundError("No datasets found. Check your dataset/ folder.")

# Combine all
df = pd.concat(datasets, ignore_index=True)

print(f"\nTotal articles (before dedup) : {len(df)}")
print(f"Fake : {len(df[df['label'] == 0])}")
print(f"Real : {len(df[df['label'] == 1])}")

# ─────────────────────────────────────────────
# 2. PREPROCESS TEXT
# ─────────────────────────────────────────────
print("\nPreprocessing text...")

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["content"] = df["content"].str.lower().str.strip()
df["content"] = df["content"].str.replace(r"\s+", " ", regex=True)

# Drop rows with empty content
df = df[df["content"].str.len() > 10].reset_index(drop=True)

# Drop duplicates across datasets
before = len(df)
df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
print(f"Duplicates removed            : {before - len(df)}")
print(f"Total articles (after dedup)  : {len(df)}")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
print("\nVectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=5,
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

with open("model/accuracy.txt", "w") as f:
    f.write(f"{accuracy * 100:.2f}")

print(f"\nModel saved      → model/model.pkl")
print(f"Vectorizer saved → model/vectorizer.pkl")
print(f"\nDone! Final accuracy: {accuracy * 100:.2f}%")