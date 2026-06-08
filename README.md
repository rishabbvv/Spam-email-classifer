<<<<<<< HEAD
# PyTorch Email Spam Classifier

This is a small beginner-friendly spam classifier built with PyTorch. It trains
on a tiny built-in dataset of spam and normal messages, then predicts whether a
new email/message is spam.

## Install
=======
# Small PyTorch Spam Email Generator

This project contains a tiny character-level GRU that learns from a short embedded
spam-like email corpus and generates new synthetic examples.

## Run

```powershell
python spam_generator.py
```

Useful options:

```powershell
python spam_generator.py --epochs 80 --temperature 0.9 --seeds claim urgent bonus
```

Install PyTorch first if it is not already available:
>>>>>>> 64e55f5 (Add PyTorch spam email generator)

```powershell
pip install torch
```

<<<<<<< HEAD
## Run

Train the model and classify a message interactively:

```powershell
python spam_classifier.py
```

Classify a message directly from the command line:

```powershell
python spam_classifier.py --email "Congratulations you won a free prize click now"
```

After the first run, the script saves a model file named `spam_classifier.pt`.
You can load it again without retraining:

```powershell
python spam_classifier.py --load spam_classifier.pt --email "Please review the assignment notes"
```

## What It Does

- Converts email text into word-count features.
- Trains a small neural network using PyTorch.
- Outputs `SPAM` or `NOT SPAM`.
- Shows the spam probability.

This dataset is intentionally small for learning. For a real spam detector, use
a larger labeled dataset and test it on emails the model has never seen.
=======
The generated text is synthetic and only intended for learning, testing, or data
augmentation experiments.
>>>>>>> 64e55f5 (Add PyTorch spam email generator)
