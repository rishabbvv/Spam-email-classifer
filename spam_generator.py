import argparse
import random
import string

import torch
from torch import nn


SPAM_EMAILS = [
    "claim your free prize now click the link to verify your account",
    "urgent winner selected send your details to receive cash today",
    "limited offer buy now and get double rewards instantly",
    "congratulations you have won a gift card reply to claim",
    "your account has a bonus waiting confirm your email now",
    "exclusive deal expires tonight open this message for savings",
    "you are pre approved for fast cash apply today",
    "verify your payment information to unlock your reward",
    "act now this special offer is only available today",
    "free vacation package waiting for you claim immediately",
    "final notice your reward will expire click now",
    "earn money from home with this simple secret system",
    "special promotion for selected users claim your coupon",
    "your invoice refund is ready provide account details",
    "security alert confirm your login to keep access",
]


class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        return self.fc(output), hidden


def build_dataset(text, stoi, seq_len):
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    xs = []
    ys = []

    for i in range(0, len(ids) - seq_len):
        xs.append(ids[i : i + seq_len])
        ys.append(ids[i + 1 : i + seq_len + 1])

    return torch.stack(xs), torch.stack(ys)


def sample(model, seed, stoi, itos, length=180, temperature=0.8, device="cpu"):
    model.eval()
    generated = seed.lower()
    hidden = None

    with torch.no_grad():
        for char in seed[:-1]:
            if char in stoi:
                x = torch.tensor([[stoi[char]]], device=device)
                _, hidden = model(x, hidden)

        current = seed[-1] if seed[-1] in stoi else random.choice(list(stoi))

        for _ in range(length):
            x = torch.tensor([[stoi[current]]], device=device)
            logits, hidden = model(x, hidden)
            logits = logits[0, -1] / max(temperature, 0.1)
            probs = torch.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, num_samples=1).item()
            current = itos[next_id]
            generated += current

    return generated.strip()


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    alphabet = string.ascii_lowercase + string.digits + " .,!?$:\n"
    corpus = "\n".join(SPAM_EMAILS).lower()
    vocab = sorted(set(corpus) | set(alphabet))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = CharRNN(len(vocab), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train, y_train = build_dataset(corpus, stoi, args.seq_len)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(x_train.size(0), device=device)
        total_loss = 0.0

        for start in range(0, x_train.size(0), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, len(vocab)), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % args.print_every == 0:
            avg_loss = total_loss / max(1, x_train.size(0) // args.batch_size)
            print(f"epoch {epoch:03d} loss {avg_loss:.4f}")

    print("\nGenerated examples:")
    for seed in args.seeds:
        print("-" * 60)
        print(sample(model, seed, stoi, itos, args.length, args.temperature, device))


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny PyTorch character spam-email generator.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--length", type=int, default=160)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["claim", "urgent", "limited"],
        help="Prompt text used to start generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
