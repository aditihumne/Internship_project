# Sentiment Analysis Streamlit App

A simple Streamlit frontend for the supplied sentiment-analysis notebook.

## Project contents

- `app.py` - Streamlit UI + prediction logic
- `backend_notebook.ipynb` - your original backend notebook
- `data/` - the attached Amazon, IMDb, and Yelp sentiment dataset
- `requirements.txt` - Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On the first run, the app trains the same SimpleRNN architecture used in the notebook and caches the trained model for the Streamlit session/runtime.

## Prediction

Type a sentence/review in the Prediction section and click **Predict sentiment**. The app shows:

- Positive or Negative prediction
- Positive-class score
- Prediction confidence
