from pathlib import Path
from contextlib import redirect_stdout
import io

import streamlit as st

from spam_classifier import load_model, predict, save_model, train_model


MODEL_PATH = Path("spam_classifier.pt")


@st.cache_resource
def get_classifier():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    training_log = io.StringIO()
    with redirect_stdout(training_log):
        model, vocab = train_model(epochs=120, lr=0.05, seed=7)
    save_model(MODEL_PATH, model, vocab)
    return model, vocab


st.set_page_config(page_title="Email Spam Classifier", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.82), rgba(20, 184, 166, 0.72)),
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.5), transparent 34%),
                linear-gradient(135deg, #eff6ff 0%, #f8fafc 48%, #ecfeff 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 900px;
            padding-top: 56px;
        }

        .hero-card,
        .input-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.62);
            border-radius: 8px;
            padding: 32px;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
        }

        .input-card {
            margin-top: 18px;
            padding: 24px;
        }

        .hero-card h1 {
            color: #172033;
            margin-bottom: 8px;
            font-size: 38px;
            line-height: 1.15;
        }

        .hero-card p {
            color: #5d6b82;
            line-height: 1.6;
            margin-bottom: 0;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
        }

        div.stButton > button {
            border-radius: 8px;
            background: #2563eb;
            color: #ffffff;
            font-weight: 700;
            border: 0;
        }

        div.stButton > button:hover {
            background: #1d4ed8;
            color: #ffffff;
            border: 0;
        }

        .hint {
            color: #5d6b82;
            font-size: 14px;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>Email Spam Classifier</h1>
        <p>Paste an email or message and check whether the model thinks it is spam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

examples = {
    "Spam example": "Congratulations you won a free prize. Click now to claim your reward.",
    "Normal example": "Please review the assignment notes and send your feedback by Friday.",
}

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

st.markdown('<div class="input-card">', unsafe_allow_html=True)

example_choice = st.selectbox("Try an example", ["Write my own"] + list(examples))
if example_choice in examples:
    st.session_state.email_text = examples[example_choice]

email_text = st.text_area(
    "Email/message text",
    key="email_text",
    placeholder="Example: Congratulations you won a free prize click now",
    height=190,
)

st.markdown('<p class="hint">The model returns a label and a spam probability.</p>', unsafe_allow_html=True)

if st.button("Check Message", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter an email or message first.")
    else:
        with st.spinner("Checking message..."):
            classifier, classifier_vocab = get_classifier()
            label, probability = predict(classifier, classifier_vocab, email_text.strip())

        if label == "SPAM":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM")

        st.metric("Spam probability", f"{probability:.2%}")

st.markdown("</div>", unsafe_allow_html=True)
