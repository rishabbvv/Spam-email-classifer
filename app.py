from pathlib import Path

import streamlit as st

from spam_classifier import load_model, predict, save_model, train_model


MODEL_PATH = Path("spam_classifier.pt")


@st.cache_resource
def get_classifier():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    model, vocab = train_model(epochs=120, lr=0.05, seed=7)
    save_model(MODEL_PATH, model, vocab)
    return model, vocab


st.set_page_config(page_title="Email Spam Classifier", page_icon="MAIL", layout="centered")

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

        .main-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.62);
            border-radius: 8px;
            padding: 32px;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.18);
        }

        .main-card h1 {
            color: #172033;
            margin-bottom: 8px;
        }

        .main-card p {
            color: #5d6b82;
            line-height: 1.6;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 8px;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-card">
        <h1>Email Spam Classifier</h1>
        <p>Paste an email or message and check whether the model thinks it is spam.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

email_text = st.text_area(
    "Email/message text",
    placeholder="Example: Congratulations you won a free prize click now",
    height=190,
)

if st.button("Check Message", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter an email or message first.")
    else:
        with st.spinner("Checking message..."):
            classifier, classifier_vocab = get_classifier()
            label, probability = predict(classifier, classifier_vocab, email_text.strip())

        if label == "SPAM":
            st.error(f"Prediction: SPAM | Spam probability: {probability:.2%}")
        else:
            st.success(f"Prediction: NOT SPAM | Spam probability: {probability:.2%}")
