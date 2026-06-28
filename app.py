import streamlit as st
import joblib

# =========================
# Load Model & Vectorizer
# =========================
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("🧠 AI powered by Machine Learning (TF-IDF + Naive Bayes)")

# =========================
# Session State
# =========================
if "text" not in st.session_state:
    st.session_state.text = ""

# =========================
# Example Button
# =========================
def load_example():
    st.session_state.text = "The movie was amazing, the acting was brilliant and the story was very engaging."

st.button("💡 Load Example", on_click=load_example)

# =========================
# Input
# =========================
text = st.text_area(
    "🎥 Enter your movie review:",
    key="text",
    height=150
)

# =========================
# Prediction
# =========================
if st.button("🔍 Analyze Sentiment"):

    if st.session_state.text.strip() == "":
        st.warning("⚠ Please enter a movie review.")
    else:

        vector = tfidf.transform([st.session_state.text])
        pred = model.predict(vector)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]

        st.markdown("---")

        # Result
        if pred == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        # Confidence
        if proba is not None:
            st.subheader("📊 Confidence Score")
            st.write(f"✔ Positive: {proba[1]*100:.2f}%")
            st.write(f"❌ Negative: {proba[0]*100:.2f}%")

        # Insight
        st.subheader("🧠 Model Insight")

        if proba is not None:
            conf = max(proba)
            if conf > 0.85:
                st.write("🔥 Very strong sentiment detected.")
            elif conf > 0.60:
                st.write("⚖ Moderate confidence prediction.")
            else:
                st.write("⚠ Low confidence prediction.")
