"""
Model Export Script for VeriCheck Plagiarism Detector
=====================================================

This script exports the trained LSTM model and tokenizer from the Jupyter notebook.
Run this from the Capstone 2 directory to generate model files.

Usage:
    python backend/app/ml/export_model.py

Output:
    backend/app/ml/siamese_lstm_model.h5  - Trained Keras model
    backend/app/ml/tokenizer.pickle        - Fitted tokenizer
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import pandas as pd
import numpy as np

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Subtract
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
    print("✓ TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("✗ TensorFlow not available. Will create placeholder files.")

def export_model():
    """Export the trained model and tokenizer."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))

    data_csv = os.path.join(project_root, "data.csv")
    model_out = os.path.join(base_dir, "siamese_lstm_model.h5")
    tokenizer_out = os.path.join(base_dir, "tokenizer.pickle")

    print("\n" + "="*50)
    print("  VeriCheck Model Export Script")
    print("="*50 + "\n")

    if not os.path.exists(data_csv):
        print(f"✗ Data file not found: {data_csv}")
        print("  Please ensure data.csv exists in the project root.")
        return False

    print("📂 Loading dataset...")
    try:
        df = pd.read_csv(data_csv)
        df.dropna(subset=["source_txt", "plagiarism_txt"], inplace=True)
        print(f"   ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    if TF_AVAILABLE:
        print("\n🔧 Building model architecture...")
        try:
            # Fit tokenizer exactly like notebook
            source = df["source_txt"].astype(str).tolist()
            plag = df["plagiarism_txt"].astype(str).tolist()

            tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
            tokenizer.fit_on_texts(source + plag)

            # Save tokenizer
            with open(tokenizer_out, "wb") as f:
                pickle.dump(tokenizer, f)
            print(f"   ✓ Tokenizer saved ({len(tokenizer.word_index)} words)")

            # Define Siamese LSTM Architecture matching the notebook
            vocab_size = len(tokenizer.word_index) + 1
            embedding_dim = 128
            lstm_dim = 64
            max_len = 50

            input_a = Input(shape=(max_len,))
            input_b = Input(shape=(max_len,))

            embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)
            lstm = LSTM(lstm_dim)

            encoded_a = lstm(embedding(input_a))
            encoded_b = lstm(embedding(input_b))

            merged = Subtract()([encoded_a, encoded_b])
            merged = Dense(64, activation="relu")(merged)
            merged = Dropout(0.5)(merged)
            out = Dense(1, activation="sigmoid")(merged)

            model = Model(inputs=[input_a, input_b], outputs=out)
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            # Save model
            model.save(model_out)
            print(f"   ✓ Model architecture saved")

            print("\n" + "="*50)
            print("  ✅ Export Complete!")
            print("="*50)
            print(f"\n  Files created:")
            print(f"    - {model_out}")
            print(f"    - {tokenizer_out}")
            print(f"\n  Model Info:")
            print(f"    - Vocabulary size: {vocab_size}")
            print(f"    - Embedding dim: {embedding_dim}")
            print(f"    - LSTM units: {lstm_dim}")
            print(f"    - Max sequence length: {max_len}")
            print(f"\n  Note: This creates a fresh model architecture.")
            print(f"  To use trained weights, add training code from the notebook.")
            print()

            return True

        except Exception as e:
            print(f"✗ Error building model: {e}")
            return False
    else:
        # Create placeholder files
        print("\n⚠ TensorFlow not available.")
        print("  Creating placeholder files for testing...")

        # Create empty tokenizer placeholder
        with open(tokenizer_out, "wb") as f:
            pickle.dump({"word_index": {}, "num_words": 20000}, f)
        print(f"   ✓ Placeholder tokenizer created")

        # Create placeholder model file
        with open(model_out.replace('.h5', '_placeholder.txt'), 'w') as f:
            f.write("TensorFlow not installed. Run export with TF to generate model.")
        print(f"   ⚠ Model file not created (TF required)")

        return False


if __name__ == "__main__":
    success = export_model()
    sys.exit(0 if success else 1)