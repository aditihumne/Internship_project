from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

st.set_page_config(page_title="Sentiment Predictor", page_icon="💬", layout="centered")

DATA_DIR = Path(__file__).parent / "data"
MAX_WORDS = 5000
MAX_LENGTH = 40


def clean_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"[^a-zA-Z0-9\s']", "", text)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    files = [
        "amazon_cells_labelled.txt",
        "imdb_labelled.txt",
        "yelp_labelled.txt",
    ]
    frames = []
    for filename in files:
        frame = pd.read_csv(
            DATA_DIR / filename,
            sep="\t",
            header=None,
            names=["sentence", "label"],
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["sentence"] = df["sentence"].astype(str).apply(clean_text)
    return df


@st.cache_resource(show_spinner="Training the sentiment model for the first run...")
def train_model():
    # This follows the backend logic from the supplied notebook.
    df = load_dataset()
    X = df["sentence"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    model = Sequential(
        [
            Embedding(input_dim=MAX_WORDS, output_dim=64, mask_zero=True),
            SimpleRNN(units=64),
            Dropout(0.3),
            Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_pad,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )
    _, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    return model, tokenizer, float(test_accuracy)


def predict_sentiment(text: str, model, tokenizer):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if score >= 0.5 else 1.0 - score
    return label, score, confidence


st.title("💬 Sentiment Analysis")
st.caption("Simple Streamlit frontend for the supplied SimpleRNN sentiment-analysis backend.")

try:
    model, tokenizer, test_accuracy = train_model()
except Exception as exc:
    st.error("The model could not be initialized.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write("SimpleRNN binary sentiment classifier")
    st.metric("Test accuracy", f"{test_accuracy * 100:.1f}%")
    st.write("Dataset: Amazon + IMDb + Yelp labeled sentences")

st.subheader("Prediction")
user_text = st.text_area(
    "Enter a review or sentence",
    placeholder="Example: I really loved this movie",
    height=130,
)

if st.button("Predict sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, score, confidence = predict_sentiment(user_text, model, tokenizer)
        if label == "Positive":
            st.success(f"Prediction: {label} 😊")
        else:
            st.error(f"Prediction: {label} 😞")

        col1, col2 = st.columns(2)
        col1.metric("Positive score", f"{score:.3f}")
        col2.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

with st.expander("Dataset preview"):
    df = load_dataset()
    st.write(f"Rows: {len(df):,}")
    st.dataframe(df.head(20), use_container_width=True)
