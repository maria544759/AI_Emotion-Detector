import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Load Model
# ----------------------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Emotion Detector")
st.write("Type any sentence below and the AI will automatically detect the emotion.")

# ----------------------------
# Advice Dictionary
# ----------------------------
advice = {
    "joy": "😊 Keep smiling! Share your happiness with someone today.",
    "sadness": "💙 It's okay to feel sad sometimes. Take a short break, listen to music, or talk to someone you trust.",
    "anger": "😌 Take a deep breath before reacting. A few calm minutes can make a big difference.",
    "fear": "💪 You're stronger than you think. Take one small step at a time.",
    "love": "❤️ Treasure the people you care about and let them know how much they mean to you.",
    "surprise": "😄 Life is full of unexpected moments. Stay curious and enjoy the adventure!"
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

# ----------------------------
# Prediction History
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_area(
    "✍️ Enter your sentence:",
    placeholder="Example: I am so happy today!"
)

# ----------------------------
# Automatic Prediction
# ----------------------------
if user_input.strip():

    vector = vectorizer.transform([user_input])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    emotion = label_encoder[prediction]
    confidence = probabilities.max()

    # Emotion
    st.subheader(f"😊 Emotion: {emotion.capitalize()}")

    # Confidence
    st.subheader("📈 Confidence")
    st.progress(int(confidence * 100))
    st.write(f"**{confidence:.2%}**")

    # Advice
    st.subheader("💡 AI Advice")
    st.info(advice.get(emotion, "Stay positive and take care of yourself."))

    # Explanation
    st.subheader("🧠 AI Explanation")
    st.write(explanation.get(emotion, ""))

    # Emotion Scores
    st.subheader("📊 Emotion Scores")

   for label, score in zip(label_encoder, probabilities):
        emoji = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲"
        }.get(label, "🙂")

        st.write(f"{emoji} **{label.capitalize()}**")
        st.progress(int(score * 100))

    # Save History
    result = {
        "Sentence": user_input,
        "Emotion": emotion.capitalize(),
        "Confidence": f"{confidence:.2%}"
    }

    if len(st.session_state.history) == 0 or st.session_state.history[-1]["Sentence"] != user_input:
        st.session_state.history.append(result)

# ----------------------------
# Prediction History
# ----------------------------
if st.session_state.history:

    st.subheader("📜 Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.table(df)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Made with ❤️ by Maria Hassan Mohamed | AI Emotion Detector | 2026")