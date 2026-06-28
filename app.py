import streamlit as st
import joblib

# =========================
# Load Model & Vectorizer
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_assets()

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# =========================
# UI Header
# =========================
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("### 🧠 AI powered by Machine Learning (TF-IDF + Naive Bayes)")

st.markdown("---")

# =========================
# Input Section
# =========================
text = st.text_area("🎥 Enter your movie review:")

# Example button
if st.button("💡 Load Example"):
    text = "The movie was amazing, the acting was brilliant and I loved it!"

# =========================
# Predict
# =========================
if st.button("🔍 Analyze"):

    if text.strip() == "":
        st.warning("⚠ Please enter a movie review.")
    else:

        # Transform text
        vector = tfidf.transform([text])

        # Prediction
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        negative = proba[0] * 100
        positive = proba[1] * 100

        st.markdown("---")

        # =========================
        # Result
        # =========================
        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        # Confidence
        st.subheader("📊 Confidence Score")
        st.progress(int(max(positive, negative)))
        st.write(f"✔ Positive: {positive:.2f}%")
        st.write(f"❌ Negative: {negative:.2f}%")

        # =========================
        # Insight
        # =========================
        st.subheader("🧠 Model Insight")

        if max(positive, negative) > 80:
            st.write("🔥 Very strong sentiment detected.")
        elif max(positive, negative) > 60:
            st.write("⚖ Moderate confidence prediction.")
        else:
            st.write("⚠ Low confidence / mixed sentiment.")

        # =========================
        # Simple keyword check
        # =========================
        st.subheader("🔍 Key Indicators")

        positive_words = ["good", "great", "amazing", "excellent", "love", "brilliant", "perfect", "awesome"]
        negative_words = ["bad", "worst", "boring", "terrible", "awful", "hate", "poor"]

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
st.caption("Built with ❤️ using Streamlit | NLP | Machine Learning")
