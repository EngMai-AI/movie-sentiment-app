import streamlit as st
from transformers import pipeline
import numpy as np

# =========================
# Load Model (RoBERTa - stronger than BERT)
# =========================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

model = load_model()

# =========================
# UI Config
# =========================
st.set_page_config(
    page_title="AI Sentiment Pro",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Advanced Sentiment Analysis AI")
st.markdown("### 🚀 Powered by RoBERTa Transformer")

# =========================
# Input
# =========================
text = st.text_area("🎥 Enter your movie review:")

# Example
if st.button("💡 Load Example"):
    text = "The movie was not perfect, but the acting was great and the story was engaging."

# =========================
# Prediction
# =========================
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter a review")
    else:

        result = model(text)[0]

        label = result["label"]
        score = result["score"]

        st.markdown("---")

        # =========================
        # Convert labels to readable
        # =========================
        if label.lower().find("positive") != -1:
            sentiment = "Positive 😊"
            color = "success"
        elif label.lower().find("negative") != -1:
            sentiment = "Negative 😞"
            color = "error"
        else:
            sentiment = "Neutral 😐"
            color = "info"

        # =========================
        # Display result
        # =========================
        if color == "success":
            st.success(sentiment)
        elif color == "error":
            st.error(sentiment)
        else:
            st.info(sentiment)

        # Confidence bar
        st.subheader("Confidence Score")
        st.progress(int(score * 100))
        st.write(f"{score*100:.2f}%")

        # =========================
        # Explanation (Simple AI reasoning)
        # =========================
        st.subheader("🧠 Model Insight")

        if score > 0.85:
            st.write("🔥 Very strong sentiment detected. Model is highly confident.")
        elif score > 0.60:
            st.write("⚖ Moderate confidence. Mixed sentiment detected.")
        else:
            st.write("⚠ Low confidence. Review may be neutral or ambiguous.")

        # =========================
        # Simple keyword highlight simulation
        # =========================
        st.subheader("🔍 Key Indicators")

        positive_words = ["good", "great", "amazing", "excellent", "love", "brilliant", "perfect"]
        negative_words = ["bad", "worst", "boring", "terrible", "awful", "hate"]

        words = text.lower().split()

        pos_found = [w for w in words if w in positive_words]
        neg_found = [w for w in words if w in negative_words]

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Positive cues: {pos_found if pos_found else 'None'}")

        with col2:
            st.error(f"Negative cues: {neg_found if neg_found else 'None'}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built with ❤️ | Advanced NLP with RoBERTa Transformers")
