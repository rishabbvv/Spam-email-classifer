# Flask PyTorch Email Spam Classifier

This is a small beginner-friendly spam classifier website built with Flask and
PyTorch. It trains on a tiny built-in dataset of spam and normal messages, then
predicts whether a new email/message is spam.

## Install

```powershell
pip install -r requirements.txt
```

## Run the Website

Start the Flask app:

```powershell
python app.py
```

Then open this URL in your browser:

http://127.0.0.1:5000

Paste an email/message into the text box and click **Check Message**.

## Run from Terminal

```powershell
python spam_classifier.py --email "Congratulations you won a free prize click now"
```

## What It Does

- Converts email text into word-count features.
- Trains a small neural network using PyTorch.
- Serves a Flask web page where users can enter a message.
- Outputs `SPAM` or `NOT SPAM`.
- Shows the spam probability.

The first website run trains the model and saves `spam_classifier.pt`. Later runs
reuse that saved model.

This dataset is intentionally small for learning. For a real spam detector, use
a larger labeled dataset and test it on emails the model has never seen.
