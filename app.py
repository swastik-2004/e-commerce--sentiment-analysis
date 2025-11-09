# ============================================================
# 🎯 SENTIMENT ANALYSIS DASHBOARD (Enhanced UI + Logging)
# ============================================================
import streamlit as st
import pandas as pd
import altair as alt
import datetime
import os
import torch
from transformers import pipeline

# ============================================================
# ⚙️ MODEL LOADING
# ============================================================
MODEL_PATH = "C:\\Users\\swastik dasgupta\\Desktop\\sentiment_analysis_project\\notebooks\\model\\checkpoint_batch_10"

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=device,
        return_all_scores=True
    )
    return sentiment_analyzer

analyzer = load_model()

# ============================================================
# 🎨 STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #eeeeee;
    }
    .stTextArea textarea {
        background-color: #222;
        color: #fff;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #00adb5;
        color: white;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #007b83;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🧠 APP HEADER
# ============================================================
st.title("💬 Sentiment Analysis using DistilBERT")
st.markdown(
    "Analyze product reviews and visualize emotional tone with BERT-based NLP."
)

# ============================================================
# 📝 INPUT SECTION
# ============================================================
st.markdown("### 🗒️ Enter your review below:")
user_input = st.text_area(
    "Type your product review:",
    height=150,
    placeholder="e.g. The product was decent but not worth the price."
)

# ============================================================
# 📊 PREDICTION & VISUALIZATION
# ============================================================
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            results = analyzer(user_input[:512])[0]  # top_k=None returns all labels
            df_scores = pd.DataFrame(results)
            df_scores["score"] = df_scores["score"] * 100
            df_scores["label"] = df_scores["label"].str.capitalize()

            # Get top prediction
            top_pred = max(results, key=lambda x: x["score"])
            label = top_pred["label"]
            confidence = top_pred["score"]

        # 🎯 Display main prediction
        st.subheader(f"Prediction: **{label}**")
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")

        # 😄😐😠 Emoji-based feedback
        if label == "Positive":
            st.success("😄 Positive sentiment detected!")
        elif label == "Negative":
            st.error("😠 Negative sentiment detected!")
        else:
            st.warning("😐 Neutral sentiment detected!")

        # 📈 Show confidence chart
        st.markdown("### 📊 Confidence Distribution")
        chart = (
            alt.Chart(df_scores)
            .mark_bar(size=40)
            .encode(
                x=alt.X("label", title="Sentiment"),
                y=alt.Y("score", title="Confidence (%)"),
                color=alt.Color("label", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                         range=["#4CAF50", "#FFC107", "#F44336"])),
                tooltip=["label", alt.Tooltip("score", format=".2f")]
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        # 🧾 Log predictions
        log_data = {
            "timestamp": datetime.datetime.now(),
            "review": user_input,
            "prediction": label,
            "confidence": confidence
        }
        log_path = os.path.join("data", "predictions_log.csv")
        os.makedirs("data", exist_ok=True)

        pd.DataFrame([log_data]).to_csv(
            log_path, mode="a", header=not os.path.exists(log_path), index=False
        )

        st.success("📁 Prediction logged successfully!")

    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# ============================================================
# 📚 EXAMPLES SECTION
# ============================================================
with st.expander("💡 Example Reviews"):
    st.write("• This product is absolutely fantastic! Love it.")
    st.write("• Not worth the price. Very disappointing.")
    st.write("• It’s okay, does the job but nothing special.")

# ============================================================
# 📈 VIEW LOG BUTTON
# ============================================================
if os.path.exists("data/predictions_log.csv"):
    with st.expander("📊 View Logged Predictions"):
        df_log = pd.read_csv("data/predictions_log.csv")
        st.dataframe(df_log.tail(10))

