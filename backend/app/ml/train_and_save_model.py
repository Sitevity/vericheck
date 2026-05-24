"""
Train and export the Siamese LSTM model for VeriCheck.
Trains on the dataset, then saves model + tokenizer.
Run: python export_model.py
"""
import os, sys, pickle, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))  # goes to Capstone 2/
DATA_CSV = os.path.join(PROJECT_ROOT, "data.csv")
MODEL_OUT = os.path.join(BASE_DIR, "siamese_lstm_model.h5")
TOKENIZER_OUT = os.path.join(BASE_DIR, "tokenizer.pickle")

import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv(DATA_CSV)
df.dropna(subset=["source_txt", "plagiarism_txt"], inplace=True)
print(f"  Loaded {len(df)} samples")

print("Fitting tokenizer...")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

source = df["source_txt"].astype(str).tolist()
plag = df["plagiarism_txt"].astype(str).tolist()

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(source + plag)
print(f"  Vocabulary: {len(tokenizer.word_index)} words")

with open(TOKENIZER_OUT, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"  Tokenizer saved → {TOKENIZER_OUT}")

print("\nVectorizing sequences...")
MAX_LEN = 50
seq_a = tokenizer.texts_to_sequences(source)
seq_b = tokenizer.texts_to_sequences(plag)
X_a = pad_sequences(seq_a, maxlen=MAX_LEN)
X_b = pad_sequences(seq_b, maxlen=MAX_LEN)
y = np.ones(len(df))  # All are plagiarized pairs
print(f"  Shapes: {X_a.shape}, {X_b.shape}")

# Subsample for fast training (full dataset takes too long)
SAMPLE = 20000
idx = np.random.choice(len(df), min(SAMPLE, len(df)), replace=False)
X_a_sub = X_a[idx]
X_b_sub = X_b[idx]
y_sub = y[idx]
print(f"  Training on {len(idx)} samples")

print("\nBuilding model...")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Subtract

VOCAB = min(len(tokenizer.word_index) + 1, 20000)
EMBED = 128
LSTM_DIM = 64

input_a = Input(shape=(MAX_LEN,))
input_b = Input(shape=(MAX_LEN,))
emb = Embedding(VOCAB, EMBED, input_length=MAX_LEN)
lstm = LSTM(LSTM_DIM)

enc_a = lstm(emb(input_a))
enc_b = lstm(emb(input_b))
merged = Subtract()([enc_a, enc_b])
merged = Dense(64, activation="relu")(merged)
merged = Dropout(0.5)(merged)
out = Dense(1, activation="sigmoid")(merged)

model = Model(inputs=[input_a, input_b], outputs=out)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

print("\nTraining (3 epochs)...")
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping

model.fit(
    [X_a_sub, X_b_sub], y_sub,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)
print(f"\nTraining done in {time.time()-start:.1f}s")

print(f"\nEvaluating...")
loss, acc = model.evaluate([X_a_sub[-2000:], X_b_sub[-2000:]], y_sub[-2000:], verbose=0)
print(f"  Accuracy: {acc:.4f}")

print(f"\nSaving model → {MODEL_OUT}")
model.save(MODEL_OUT)
print("Done!")