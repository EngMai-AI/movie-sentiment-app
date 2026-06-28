import streamlit as st
from transformers import pipeline

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

model = load_model()

# =========================
# UI
# =========================
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("🧠 AI powered by RoBERTa Transformer")

text = st.text_area("🎥 Enter your movie review:")

# Example button
if st.button("💡 Try Example"):
    text = "The movie was amazing, the acting was brilliant and I loved it."

# Predict
if st.button("🔍 Analyze"):

    if text.strip() == "":
        st.warning("Please enter a review")
    else:
        result = model(text)[0]

        label = result["label"]
        score = result["score"]

        # Mapping labels
        if "positive" in label.lower():
            sentiment = "Positive 😊"
        elif "negative" in label.lower():
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        st.markdown("---")

        if sentiment.startswith("Positive"):
            st.success(sentiment)
        elif sentiment.startswith("Negative"):
            st.error(sentiment)
        else:
            st.info(sentiment)

        st.subheader("Confidence Score")
        st.progress(int(score * 100))
        st.write(f"{score*100:.2f}%")

        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit | NLP | Transformers")
