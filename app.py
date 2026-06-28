import streamlit as st
import numpy as np
import joblib

# =========================
# Load Model
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_assets()

# =========================
# UI Settings
# =========================
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("🧠 AI powered by Machine Learning (TF-IDF + Naive Bayes)")

# =========================
# Input
# =========================
text = st.text_area("🎥 Enter your movie review:")

if st.button("💡 Example"):
    text = "The movie was amazing, the acting was brilliant and the story was perfect."

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Sentiment"):

    if text.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        cleaned = text.lower()
        vector = tfidf.transform([cleaned])

        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        negative = proba[0]
        positive = proba[1]

        st.markdown("---")

        # =========================
        # Result
        # =========================
        if pred == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        # =========================
        # Confidence
        # =========================
        st.subheader("📊 Confidence Score")

        st.progress(int(max(positive, negative) * 100))

        st.write(f"✔ Positive: {positive*100:.2f}%")
        st.write(f"❌ Negative: {negative*100:.2f}%")

        # =========================
        # Insight
        # =========================
        st.subheader("🧠 Model Insight")

        if max(positive, negative) > 0.85:
            st.write("🔥 Very strong sentiment detected.")
        elif max(positive, negative) > 0.60:
            st.write("⚖ Moderate confidence prediction.")
        else:
            st.write("⚠ Weak sentiment / ambiguous review.")

        # =========================
        # Footer
        # =========================
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit | NLP | Machine Learning")
