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

```powershell
pip install torch
```

The generated text is synthetic and only intended for learning, testing, or data
augmentation experiments.
