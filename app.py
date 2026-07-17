import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown("""
<style>
    .main {
        background-color: #F7F8FA;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background-color: #EEF2FF;
        color: #4338CA;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .section-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }
    .score-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.15rem;
    }
    .stButton>button, .stFormSubmitButton>button {
        background-color: #4338CA;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #372AA8;
        color: white;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model_files():
    with open("emotion_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)

    return model, vectorizer, labels

model, vectorizer, labels = load_model_files()

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="app-title">AI Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Type a sentence and let the AI analyze the emotion behind it.</div>',
    unsafe_allow_html=True
)

# ----------------------------
# Advice Dictionary
# ----------------------------
advice = {
    "joy": "Keep smiling! Share your happiness with someone today.",
    "sadness": "It's okay to feel sad sometimes. Take a short break, listen to music, or talk to someone you trust.",
    "anger": "Take a deep breath before reacting. A few calm minutes can make a big difference.",
    "fear": "You're stronger than you think. Take one small step at a time.",
    "love": "Treasure the people you care about and let them know how much they mean to you.",
    "surprise": "Life is full of unexpected moments. Stay curious and enjoy the adventure!"
}

# ----------------------------
# Explanation Dictionary
# ----------------------------
explanation = {
    "joy": "This sentence contains positive words that usually express happiness or excitement.",
    "sadness": "This sentence contains words related to sadness or disappointment.",
    "anger": "This sentence expresses frustration or anger.",
    "fear": "This sentence contains words that indicate worry or fear.",
    "love": "This sentence expresses affection or caring feelings.",
    "surprise": "This sentence expresses unexpected feelings or amazement."
}

EMOTION_COLORS = {
    "joy": "#F59E0B",
    "sadness": "#3B82F6",
    "anger": "#EF4444",
    "fear": "#8B5CF6",
    "love": "#EC4899",
    "surprise": "#10B981",
}

# ----------------------------
# Prediction History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# User Input (wrapped in a form so pressing Enter submits it)
# ----------------------------
with st.form(key="emotion_form", border=False):
    user_input = st.text_input(
        "Enter your sentence",
        placeholder="e.g. I am so happy today!"
    )
    submitted = st.form_submit_button("Analyze")

# ----------------------------
# Prediction
# ----------------------------
if submitted:

    if user_input.strip():

        vector = vectorizer.transform([user_input])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        emotion = labels[prediction]
        confidence = probabilities.max()
        color = EMOTION_COLORS.get(emotion, "#4338CA")

        # ---- Result Card ----
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(
            f'<span class="emotion-badge" style="background-color:{color}20; color:{color};">'
            f'{emotion.capitalize()}</span>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
        st.progress(int(confidence * 100))
        st.write(f"{confidence:.1%}")

        st.markdown('<div class="section-label">Advice</div>', unsafe_allow_html=True)
        st.write(advice.get(emotion, "Stay positive and take care of yourself."))

        st.markdown('<div class="section-label">Emotion Breakdown</div>', unsafe_allow_html=True)
        for label, score in zip(labels, probabilities):
            st.markdown(
                f'<div class="score-row"><span>{label.capitalize()}</span>'
                f'<span>{score:.0%}</span></div>',
                unsafe_allow_html=True
            )
            st.progress(int(score * 100))

        st.markdown('</div>', unsafe_allow_html=True)

        # Save History
        result = {
            "Sentence": user_input,
            "Emotion": emotion.capitalize(),
            "Confidence": f"{confidence:.2%}"
        }

        if (
            len(st.session_state.history) == 0
            or st.session_state.history[-1]["Sentence"] != user_input
        ):
            st.session_state.history.append(result)

# ----------------------------
# Prediction History
# ----------------------------
if st.session_state.history:

    st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Made by Maria Hassan Mohamed · AI Emotion Detector · 2026")