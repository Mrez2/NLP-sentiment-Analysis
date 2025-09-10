import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ------------------------------
# Parameters
# ------------------------------
max_features = 10000  # number of words to keep from IMDB
maxlen = 200          # sequence length

# Load IMDB word index with offset
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}  # shift by 3
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {v: k for k, v in word_index.items()}

# Load trained model
@st.cache_resource
def load_sentiment_model():
    return load_model("model.h5")

model = load_sentiment_model()

# ------------------------------
# Helper functions
# ------------------------------
def encode_review(text):
    """Convert review text into integer sequence using IMDB word index."""
    words = text.lower().split()
    encoded = [1]  # start token
    for w in words:
        idx = word_index.get(w, 2)  # 2 = <UNK>
        encoded.append(idx)
    return encoded

def decode_review(encoded):
    """Convert integers back to words (for debug)."""
    return " ".join([reverse_word_index.get(i, "?") for i in encoded])

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎬 IMDB Sentiment Analysis with LSTM")

user_review = st.text_area("✍️ Enter a movie review for sentiment analysis:")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        # Encode review with correct offset
        encoded = encode_review(user_review)
        unk_count = encoded.count(2)
        padded = pad_sequences([encoded], maxlen=maxlen)

        # Predict
        prediction = float(model.predict(padded)[0][0])
        st.write(f"🔢 Raw score: {prediction:.4f} (0=Negative, 1=Positive)")
        st.write(f"❓ Unknown words in your review: {unk_count}")

        # Classification
        if prediction >= 0.5:
            st.success(f"🌟 Positive Review (Confidence: {prediction:.2f})")
        else:
            st.error(f"👎 Negative Review (Confidence: {1 - prediction:.2f})")

        # Debug info
        st.write("🔍 Encoded tokens (first 30):", encoded[:30])
        st.write("🔍 Decoded back:", decode_review(encoded[:30]))
