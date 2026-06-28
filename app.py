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
st.text_area(
    "🎥 Enter your movie review:",
    key="text",
    height=150
)

# =========================
# Prediction Button
# =========================
if st.button("🔍 Analyze Sentiment"):

    if st.session_state.text.strip() == "":
        st.warning("⚠ Please enter a movie review.")
    else:

        # Transform input
        vector = tfidf.transform([st.session_state.text])
        pred = model.predict(vector)[0]

        # Probability (if supported)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vector)[0]

        st.markdown("---")

        # =========================
        # Result
        # =========================
        if pred == 1:
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

        # =========================
        # Confidence Score
        # =========================
        if proba is not None:
            st.subheader("📊 Confidence Score")

            st.progress(int(proba[1] * 100))
            st.write(f"✔ Positive: {proba[1]*100:.2f}%")

            st.progress(int(proba[0] * 100))
            st.write(f"❌ Negative: {proba[0]*100:.2f}%")

        # =========================
        # Model Insight
        # =========================
        st.subheader("🧠 Model Insight")

        if proba is not None:
            confidence = max(proba)

            if confidence > 0.85:
                st.info("🔥 Very strong sentiment detected.")
            elif confidence > 0.60:
                st.warning("⚖ Moderate confidence prediction.")
            else:
                st.error("⚠ Low confidence prediction.")

        # =========================
        # Simple Keyword Indicators
        # =========================
        st.subheader("🔍 Key Indicators")

        positive_words = ["good", "great", "amazing", "excellent", "love", "brilliant", "perfect"]
        negative_words = ["bad", "worst", "boring", "terrible", "awful", "hate"]

        words = st.session_state.text.lower().split()

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
st.caption("🚀 Built with Streamlit | NLP | Machine Learning | TF-IDF + Naive Bayes")
