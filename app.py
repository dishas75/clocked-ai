import os
import pickle
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from textstat import flesch_reading_ease

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load model and encoder ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
        payload = pickle.load(f)
    with open(os.path.join(BASE, "models", "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    # extract the actual classifier from the dict
    if isinstance(payload, dict):
        model = payload.get("model") or payload.get("classifier") or list(payload.values())[0]
    else:
        model = payload
    return model, le

model, le = load_model()

# ── Feature extraction (must match extract_features.py) ───────
def extract_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    sentence_variance = pd.Series(sentence_lengths).std() if len(sentence_lengths) > 1 else 0
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / word_count if word_count else 0
    hapax_ratio = len([w for w, c in Counter(w.lower() for w in words).items() if c == 1]) / word_count if word_count else 0
    long_word_ratio = sum(1 for w in words if len(w) >= 7) / word_count if word_count else 0
    short_word_ratio = sum(1 for w in words if len(w) <= 3) / word_count if word_count else 0
    freq = Counter(w.lower() for w in words)
    repetition_ratio = max(freq.values()) / word_count if word_count else 0
    top5_word_freq = sum(c for _, c in freq.most_common(5)) / word_count if word_count else 0
    question_count = text.count("?")
    exclamation_count = text.count("!")
    comma_count = text.count(",")
    colon_count = text.count(":")
    semicolon_count = text.count(";")
    comma_rate = comma_count / word_count if word_count else 0
    colon_rate = colon_count / word_count if word_count else 0
    bullet_count = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
    numbered_list_count = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    markdown_headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
    code_blocks = text.count("```")
    has_bold = int("**" in text)
    paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
    avg_paragraph_length = word_count / paragraph_count if paragraph_count else 0
    list_density = (bullet_count + numbered_list_count) / sentence_count if sentence_count else 0
    capitalized_words = sum(1 for w in words if len(w) > 1 and w[0].isupper())
    capitalized_ratio = capitalized_words / word_count if word_count else 0
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    hedge_words = ["perhaps","might","could","may","possibly","likely","probably","generally","typically","sometimes","often"]
    hedge_rate = sum(1 for w in words if w.lower() in hedge_words) / word_count if word_count else 0
    formal_words = ["therefore","however","furthermore","moreover","thus","consequently"]
    formality_rate = sum(1 for w in words if w.lower() in formal_words) / word_count if word_count else 0
    filler_words = ["actually","basically","certainly","indeed","essentially","specifically","particularly"]
    filler_rate = sum(1 for w in words if w.lower() in filler_words) / word_count if word_count else 0
    empathy_words = ["feel","feeling","understand","support","help","care","normal","difficult","hard"]
    empathy_rate = sum(1 for w in words if w.lower() in empathy_words) / word_count if word_count else 0
    first_person = ["i","me","my","mine"]
    second_person = ["you","your","yours"]
    first_person_rate = sum(1 for w in words if w.lower() in first_person) / word_count if word_count else 0
    second_person_rate = sum(1 for w in words if w.lower() in second_person) / word_count if word_count else 0
    readability = flesch_reading_ease(text)
    starts_with_sure = int(text.lower().startswith("sure"))
    starts_with_certainly = int(text.lower().startswith("certainly"))
    starts_with_here = int(text.lower().startswith("here"))
    contains_example = int("example" in text.lower())
    contains_steps = int("step" in text.lower())
    contains_summary = int("summary" in text.lower())
    contains_conclusion = int("conclusion" in text.lower())
    parentheses_count = text.count("(") + text.count(")")
    quote_count = text.count('"') + text.count("'")

    return {
        "word_count": word_count, "char_count": char_count, "sentence_count": sentence_count,
        "avg_word_length": avg_word_length, "avg_sentence_length": avg_sentence_length,
        "sentence_variance": sentence_variance, "type_token_ratio": type_token_ratio,
        "hapax_ratio": hapax_ratio, "long_word_ratio": long_word_ratio,
        "short_word_ratio": short_word_ratio, "repetition_ratio": repetition_ratio,
        "top5_word_freq": top5_word_freq, "question_count": question_count,
        "exclamation_count": exclamation_count, "comma_count": comma_count,
        "colon_count": colon_count, "semicolon_count": semicolon_count,
        "comma_rate": comma_rate, "colon_rate": colon_rate, "bullet_count": bullet_count,
        "numbered_list_count": numbered_list_count, "markdown_headers": markdown_headers,
        "code_blocks": code_blocks, "has_bold": has_bold, "paragraph_count": paragraph_count,
        "avg_paragraph_length": avg_paragraph_length, "list_density": list_density,
        "capitalized_ratio": capitalized_ratio, "all_caps_words": all_caps_words,
        "hedge_rate": hedge_rate, "formality_rate": formality_rate, "filler_rate": filler_rate,
        "empathy_rate": empathy_rate, "first_person_rate": first_person_rate,
        "second_person_rate": second_person_rate, "readability": readability,
        "starts_with_sure": starts_with_sure, "starts_with_certainly": starts_with_certainly,
        "starts_with_here": starts_with_here, "contains_example": contains_example,
        "contains_steps": contains_steps, "contains_summary": contains_summary,
        "contains_conclusion": contains_conclusion, "parentheses_count": parentheses_count,
        "quote_count": quote_count
    }

# ── Display names ──────────────────────────────────────────────
MODEL_DISPLAY = {
    "gemma-4-26b-a4b-it":        "Gemma 4 26B",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B",
    "openai/gpt-oss-120b":     "GPT OSS 120B",
    "qwen/qwen3-32b":          "Qwen 3 32B",
}

MODEL_COLORS = {
    "gemma-4-26b-a4b-it":        "#4285F4",
    "llama-3.3-70b-versatile": "#7C3AED",
    "openai/gpt-oss-120b":     "#10A37F",
    "qwen/qwen3-32b":          "#F59E0B",
}

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Clocked — AI Linguistic Forensics",
    page_icon="🕵️",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; font-size:2.8rem;'>🕵️ Clocked</h1>
    <p style='text-align:center; color:gray; font-size:1.1rem;'>
        Every model leaves a signature. Can you hide yours?
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Fingerprint Text", "📊 Model Insights", "ℹ️ About"])

# ── Tab 1: Predict ─────────────────────────────────────────────
with tab1:
    st.subheader("Paste any AI-generated text below")
    text_input = st.text_area(
        "Text to analyze",
        height=250,
        placeholder="Paste a response from any LLM here..."
    )

    if st.button("🔍 Identify Model", use_container_width=True):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            features = extract_features(text_input)
            if features is None:
                st.error("Could not extract features. Try a longer text.")
            else:
                X_input = pd.DataFrame([features])
                proba = model.predict_proba(X_input)[0]
                pred_idx = np.argmax(proba)
                pred_label = le.classes_[pred_idx]
                confidence = proba[pred_idx] * 100

                display_name = MODEL_DISPLAY.get(pred_label, pred_label)
                color = MODEL_COLORS.get(pred_label, "#888")

                st.markdown(f"""
                    <div style='text-align:center; padding:2rem; border-radius:12px;
                                background:{color}20; border: 2px solid {color};'>
                        <h2 style='color:{color};'>Most Likely: {display_name}</h2>
                        <h3>Confidence: {confidence:.1f}%</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Confidence scores across all models")
                conf_df = pd.DataFrame({
                    "Model": [MODEL_DISPLAY.get(le.classes_[i], le.classes_[i]) for i in range(len(le.classes_))],
                    "Confidence (%)": [p * 100 for p in proba]
                }).sort_values("Confidence (%)", ascending=False)

                st.bar_chart(conf_df.set_index("Model"))

                st.markdown("#### Detected linguistic traits")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Word Count", features["word_count"])
                col2.metric("Avg Sentence Length", f"{features['avg_sentence_length']:.1f}")
                col3.metric("Bullet Points", features["bullet_count"])
                col4.metric("Hedge Rate", f"{features['hedge_rate']:.3f}")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Type-Token Ratio", f"{features['type_token_ratio']:.3f}")
                col6.metric("Formality Rate", f"{features['formality_rate']:.3f}")
                col7.metric("Readability", f"{features['readability']:.1f}")
                col8.metric("Paragraph Count", features["paragraph_count"])

# ── Tab 2: Model Insights ──────────────────────────────────────
with tab2:
    st.subheader("How well does Clocked perform?")

    col1, col2 = st.columns(2)
    with col1:
        cm_path = os.path.join(BASE, "models", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)

    with col2:
        fi_path = os.path.join(BASE, "models", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance", use_container_width=True)

    st.subheader("Model Comparison")
    mc_path = os.path.join(BASE, "models", "model_comparison.png")
    if os.path.exists(mc_path):
        st.image(mc_path, caption="All classifiers compared", use_container_width=True)

    st.subheader("SHAP Analysis")
    shap_bar = os.path.join(BASE, "models", "shap", "shap_bar.png")
    shap_summary = os.path.join(BASE, "models", "shap", "shap_summary.png")

    if os.path.exists(shap_bar):
        st.image(shap_bar, caption="SHAP Feature Importance (all classes)", use_container_width=True)
    if os.path.exists(shap_summary):
        st.image(shap_summary, caption="SHAP Beeswarm Summary", use_container_width=True)

    st.subheader("Per-model SHAP breakdown")
    shap_cols = st.columns(2)
    for i, (model_id, display_name) in enumerate(MODEL_DISPLAY.items()):
        safe_name = model_id.replace("/", "_").replace(".", "_")
        path = os.path.join(BASE, "models", "shap", f"shap_{safe_name}.png")
        if os.path.exists(path):
            shap_cols[i % 2].image(path, caption=f"SHAP — {display_name}", use_container_width=True)

# ── Tab 3: About ───────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## 🕵️ Clocked — AI Linguistic Forensics

    **Clocked** identifies which large language model wrote a piece of text
    using stylometric fingerprinting alone — no metadata, no watermarks, just the words.

    ### Models studied
    | Model | Provider |
    |---|---|
    | Gemma 4 26B | Google |
    | LLaMA 3.3 70B | Meta |
    | GPT OSS 120B | OpenAI |
    | Qwen 3 32B | Alibaba |

    ### How it works
    1. Extract 44 linguistic features from the text
    2. Feed into a Random Forest classifier (93.3% accuracy)
    3. Return prediction + confidence scores + feature breakdown

    ### Top fingerprint features
    - Sentence count and length distribution
    - Type-token ratio (vocabulary diversity)
    - Bullet point and list usage
    - Hedge word frequency
    - Paragraph structure

    ### Built by
    **Disha S** — as part of OpenCode Summer Task 2026.
    """)