"""
Train Siamese LSTM on real SNLI-style pairs with proper positive/negative labeling.
Uses source_txt as anchor; positive = plagiarism_txt with same topic, negative = random sentence.
"""
import os, pickle, time
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
MODEL_OUT    = os.path.join(BASE_DIR, "siamese_lstm_model.h5")
TOKEN_OUT    = os.path.join(BASE_DIR, "tokenizer.pickle")
DATA_CSV     = os.path.join(PROJECT_ROOT, "data.csv")

MAX_LEN    = 50
VOCAB_SIZE = 20000
EMBED_DIM  = 128
LSTM_DIM   = 64
EPOCHS     = 10
BATCH      = 256
N_POS      = 15000
N_NEG      = 15000

np.random.seed(42)

print("=" * 55)
print("  VeriCheck — SNLI-based Model Training")
print("=" * 55)

# ── Load Data ────────────────────────────────────────────────────────────────
print("\n📂 Loading SNLI dataset...")
df = pd.read_csv(DATA_CSV)
df.dropna(subset=["source_txt", "plagiarism_txt"])
df["source_txt"]     = df["source_txt"].astype(str).str.strip()
df["plagiarism_txt"] = df["plagiarism_txt"].astype(str).str.strip()
df["label"]          = df["label"].astype(int)

# Filter to reasonable length (15–200 chars)
df = df[(df["source_txt"].str.len() > 15) & (df["source_txt"].str.len() < 200)]
df = df[(df["plagiarism_txt"].str.len() > 15) & (df["plagiarism_txt"].str.len() < 200)].reset_index(drop=True)
print(f"   Rows after filtering: {len(df)}")

# ── Build Positive Pairs (plagiarized = paraphrase of source) ───────────────
pos_df = df.sample(n=min(N_POS, len(df)), random_state=42)
pos_a  = pos_df["source_txt"].tolist()
pos_b  = pos_df["plagiarism_txt"].tolist()
print(f"   Positive pairs: {len(pos_a)}")

# ── Build Negative Pairs (source vs different random source) ─────────────────
neg_a = pos_df["source_txt"].tolist()
# Pick random different sentences for negative pairs
neg_pool = df["source_txt"].tolist() + df["plagiarism_txt"].tolist()
np.random.seed(42)
neg_b = [neg_pool[np.random.randint(0, len(neg_pool))] for _ in range(len(pos_a))]
print(f"   Negative pairs: {len(neg_a)}")

# ── Combine & Shuffle ─────────────────────────────────────────────────────────
all_a = pos_a + neg_a
all_b = pos_b + neg_b
all_y = [1.0] * len(pos_a) + [0.0] * len(neg_a)

perm = np.random.permutation(len(all_y))
all_a = [all_a[i] for i in perm]
all_b = [all_b[i] for i in perm]
all_y = np.array([all_y[i] for i in perm], dtype=np.float32)

# ── Tokenize ───────────────────────────────────────────────────────────────────
print("\n🔤 Building tokenizer...")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(all_a + all_b)
print(f"   Vocabulary: {len(tokenizer.word_index)} words")

with open(TOKEN_OUT, "wb") as f:
    pickle.dump(tokenizer, f)

def vectorize(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post")

Xa = vectorize(all_a)
Xb = vectorize(all_b)
y  = all_y

print(f"   Xa: {Xa.shape}  Xb: {Xb.shape}")
print(f"   y=1: {(y==1).sum()}  y=0: {(y==0).sum()}")

# ── Split ──────────────────────────────────────────────────────────────────────
split = int(len(y) * 0.85)
Xa_tr, Xa_te = Xa[:split], Xa[split:]
Xb_tr, Xb_te = Xb[:split], Xb[split:]
y_tr,  y_te  = y[:split],   y[split:]
print(f"   Train: {len(y_tr)}  |  Test: {len(y_te)}")

# ── Model ──────────────────────────────────────────────────────────────────────
print("\n🏗️  Building model...")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Subtract, Multiply, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

input_a = Input(shape=(MAX_LEN,), name="in_a")
input_b = Input(shape=(MAX_LEN,), name="in_b")

shared_emb  = Embedding(VOCAB_SIZE, EMBED_DIM, name="emb")
shared_lstm = LSTM(LSTM_DIM, return_sequences=False, name="lstm")

ea = shared_lstm(shared_emb(input_a))
eb = shared_lstm(shared_emb(input_b))

diff     = Subtract()([ea, eb])
prod     = Multiply()([ea, eb])
combined = Add()([diff, prod])

d1 = Dense(128, activation="relu")(combined)
d1 = Dropout(0.4)(d1)
d2 = Dense(64,  activation="relu")(d1)
d2 = Dropout(0.4)(d2)
out = Dense(1,  activation="sigmoid")(d2)

model = Model(inputs=[input_a, input_b], outputs=out)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()

# ── Train ──────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

print("\n🚀 Training...")
t0 = time.time()
model.fit([Xa_tr, Xb_tr], y_tr,
          validation_data=([Xa_te, Xb_te], y_te),
          epochs=EPOCHS, batch_size=BATCH,
          callbacks=callbacks, verbose=1)
print(f"   Done in {time.time()-t0:.0f}s")

# ── Evaluate ────────────────────────────────────────────────────────────────────
print("\n📊 Test Evaluation:")
loss, acc, auc = model.evaluate([Xa_te, Xb_te], y_te, verbose=0)
preds = (model.predict([Xa_te, Xb_te], verbose=0).flatten() > 0.5).astype(int)
tp = ((preds==1)&(y_te==1)).sum()
tn = ((preds==0)&(y_te==0)).sum()
fp = ((preds==1)&(y_te==0)).sum()
fn = ((preds==0)&(y_te==1)).sum()
prec = tp/(tp+fp+1e-10)
rec  = tp/(tp+fn+1e-10)
f1   = 2*prec*rec/(prec+rec+1e-10)
print(f"   Accuracy:  {acc:.4f} ({acc:.1%})")
print(f"   AUC:       {auc:.4f} ({auc:.1%})")
print(f"   Precision: {prec:.4f} ({prec:.1%})")
print(f"   Recall:    {rec:.4f} ({rec:.1%})")
print(f"   F1:        {f1:.4f} ({f1:.1%})")
print(f"   Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

print(f"\n💾 Saving → {MODEL_OUT}")
model.save(MODEL_OUT)
print("\n✅ Done!")