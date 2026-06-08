from pathlib import Path

from flask import Flask, render_template, request

from spam_classifier import load_model, predict, save_model, train_model


MODEL_PATH = Path("spam_classifier.pt")

app = Flask(__name__)
model = None
vocab = None


def get_classifier():
    global model, vocab

    if model is not None and vocab is not None:
        return model, vocab

    if MODEL_PATH.exists():
        model, vocab = load_model(MODEL_PATH)
    else:
        model, vocab = train_model(epochs=120, lr=0.05, seed=7)
        save_model(MODEL_PATH, model, vocab)

    return model, vocab


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if email_text:
            classifier, classifier_vocab = get_classifier()
            label, probability = predict(classifier, classifier_vocab, email_text)
            result = {
                "label": label,
                "probability": probability,
                "percent": f"{probability:.2%}",
                "is_spam": label == "SPAM",
            }

    return render_template("index.html", result=result, email_text=email_text)


if __name__ == "__main__":
    app.run(debug=True)
