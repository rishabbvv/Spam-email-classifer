import argparse
import re

import torch
from torch import nn


TRAINING_DATA = [
    ("claim your free prize now click the link to verify your account", 1),
    ("urgent winner selected send your details to receive cash today", 1),
    ("limited offer buy now and get double rewards instantly", 1),
    ("congratulations you have won a gift card reply to claim", 1),
    ("your account has a bonus waiting confirm your email now", 1),
    ("exclusive deal expires tonight open this message for savings", 1),
    ("you are pre approved for fast cash apply today", 1),
    ("verify your payment information to unlock your reward", 1),
    ("act now this special offer is only available today", 1),
    ("free vacation package waiting for you claim immediately", 1),
    ("final notice your reward will expire click now", 1),
    ("earn money from home with this simple secret system", 1),
    ("security alert confirm your login to keep access", 1),
    ("your invoice refund is ready provide account details", 1),
    ("special promotion for selected users claim your coupon", 1),
    ("meeting moved to three pm please review the agenda", 0),
    ("hi can you send the project report before friday", 0),
    ("your package was delivered at the front desk", 0),
    ("thanks for your email I will call you tomorrow", 0),
    ("please find the attached notes from today's class", 0),
    ("the team lunch is scheduled for next monday", 0),
    ("can we reschedule our appointment to next week", 0),
    ("your electricity bill has been paid successfully", 0),
    ("here is the document you requested yesterday", 0),
    ("mom asked if you are coming home this weekend", 0),
    ("the library book renewal is confirmed", 0),
    ("your password was changed successfully as requested", 0),
    ("please review the code changes when you have time", 0),
    ("flight booking confirmation for your upcoming trip", 0),
    ("the assignment deadline has been extended", 0),
]


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vocab_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def tokenize(text):
    return re.findall(r"[a-z0-9$]+", text.lower())


def build_vocab(samples):
    words = sorted({word for text, _ in samples for word in tokenize(text)})
    return {"<unk>": 0, **{word: idx + 1 for idx, word in enumerate(words)}}


def vectorize(text, vocab):
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    for word in tokenize(text):
        vector[vocab.get(word, vocab["<unk>"])] += 1
    total = vector.sum()
    return vector / total if total > 0 else vector


def make_tensors(samples, vocab):
    x = torch.stack([vectorize(text, vocab) for text, _ in samples])
    y = torch.tensor([label for _, label in samples], dtype=torch.float32)
    return x, y


def train_model(epochs, lr, seed):
    torch.manual_seed(seed)
    vocab = build_vocab(TRAINING_DATA)
    x_train, y_train = make_tensors(TRAINING_DATA, vocab)

    model = SpamClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 20 == 0:
            with torch.no_grad():
                predictions = (torch.sigmoid(model(x_train)) >= 0.5).float()
                accuracy = (predictions == y_train).float().mean().item()
            print(f"epoch {epoch:03d} loss {loss.item():.4f} accuracy {accuracy:.2%}")

    return model, vocab


def predict(model, vocab, email_text):
    model.eval()
    with torch.no_grad():
        features = vectorize(email_text, vocab).unsqueeze(0)
        spam_probability = torch.sigmoid(model(features)).item()
    label = "SPAM" if spam_probability >= 0.5 else "NOT SPAM"
    return label, spam_probability


def save_model(path, model, vocab):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
        },
        path,
    )


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    vocab = checkpoint["vocab"]
    model = SpamClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Small PyTorch email spam classifier.")
    parser.add_argument("--email", type=str, help="Email text to classify.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=str, default="spam_classifier.pt")
    parser.add_argument("--load", type=str, help="Load an existing saved model instead of training.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.load:
        model, vocab = load_model(args.load)
        print(f"loaded model from {args.load}")
    else:
        model, vocab = train_model(args.epochs, args.lr, args.seed)
        save_model(args.save, model, vocab)
        print(f"saved model to {args.save}")

    email_text = args.email or input("\nEnter an email/message to classify: ")
    label, probability = predict(model, vocab, email_text)
    print(f"\nPrediction: {label}")
    print(f"Spam probability: {probability:.2%}")


if __name__ == "__main__":
    main()
