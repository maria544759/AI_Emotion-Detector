import streamlit as st
import pickle

# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Load AI Model
# -------------------------------
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("labels.pkl", "rb") as f:
    id2label = pickle.load(f)

# -------------------------------
# Emojis
# -------------------------------
emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# -------------------------------
# Advice
# -------------------------------
advice = {
    "joy": "😊 Keep smiling! Share your happiness with someone today.",
    "sadness": "💙 It's okay to feel sad sometimes. Take a short break, listen to music, or talk to someone you trust.",
    "anger": "😌 Take a deep breath before reacting. A few calm minutes can make a big difference.",
    "fear": "💪 You're stronger than you think. Take one small step at a time.",
    "love": "❤️ Treasure the people you care about and let them know how much they mean to you.",
    "surprise": "😄 Life is full of unexpected moments. Stay curious and enjoy the adventure!"
}

# -------------------------------
# AI Explanation
# -------------------------------
explanation = {
    "joy": "The sentence contains positive or exciting words that usually express happiness.",
    "sadness": "The sentence contains words related to disappointment, loss, or sadness.",
    "anger": "The sentence includes words associated with frustration or anger.",
    "fear": "The sentence suggests worry, nervousness, or fear.",
    "love": "The sentence expresses affection, care, or appreciation.",
    "surprise": "The sentence contains expressions of shock or something unexpected."
}

# -------------------------------
# Background Color Function
# -------------------------------
def set_background(emotion):
    colors = {
        "joy": "#FFF9C4",
        "sadness": "#BBDEFB",
        "anger": "#FFCDD2",
        "fear": "#E1BEE7",
        "love": "#F8BBD0",
        "surprise": "#FFE0B2"
    }

    color = colors.get(emotion, "#FFFFFF")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# Prediction History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Title
# -------------------------------
st.title("🤖 AI Emotion Detector")

st.write("Type any sentence below and let AI predict the emotion.")

text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: I got a new laptop today and I'm very excited!"
)

# -------------------------------
# Detect Emotion
# -------------------------------
if st.button("🔍 Detect Emotion"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities)

        emotion = id2label[prediction]

        set_background(emotion)

        st.success(f"{emoji[emotion]} Emotion: **{emotion.upper()}**")

        st.write("### Confidence")
        st.progress(float(confidence))
        st.write(f"**{confidence*100:.2f}%**")

        st.subheader("💡 AI Advice")
        st.info(advice[emotion])

        st.subheader("🧠 AI Explanation")
        st.write(explanation[emotion])

        st.subheader("📊 Emotion Scores")

        for i, score in enumerate(probabilities):
            st.write(f"{emoji[id2label[i]]} **{id2label[i].title()}**")
            st.progress(float(score))

        # Save prediction history
        st.session_state.history.append({
            "Sentence": text,
            "Emotion": f"{emoji[emotion]} {emotion.title()}",
            "Confidence": f"{confidence*100:.2f}%"
        })

# -------------------------------
# Show History
# -------------------------------
if st.session_state.history:

    st.subheader("📜 Prediction History")

    st.table(st.session_state.history)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()