# Streamlit PyTorch Email Spam Classifier

This is a small beginner-friendly spam classifier app built with Streamlit and
PyTorch. It trains on a tiny built-in dataset of spam and normal messages, then
predicts whether a new email/message is spam.

## Install

```powershell
pip install -r requirements.txt
```

## Run the Streamlit App

```powershell
streamlit run app.py
```

Streamlit will open the app in your browser. If it does not open automatically,
use the local URL shown in the terminal.

## Run from Terminal

```powershell
python spam_classifier.py --email "Congratulations you won a free prize click now"
```

## What It Does

- Converts email text into word-count features.
- Trains a small neural network using PyTorch.
- Serves a Streamlit interface where users can enter a message.
- Outputs `SPAM` or `NOT SPAM`.
- Shows the spam probability.

The first app run trains the model and saves `spam_classifier.pt`. Later runs
reuse that saved model.

This dataset is intentionally small for learning. For a real spam detector, use
a larger labeled dataset and test it on emails the model has never seen.
