# 🕵️ Clocked

> ### *Every model leaves a signature.*

Clocked is an AI linguistic forensics system that identifies which large language model wrote a piece of text — using writing style, syntax patterns, and linguistic fingerprints alone. No metadata. No hints. Just the words, and the truth inside them.

---

## 📊 At a Glance

| Models Studied | Unique Prompts | Labeled Samples |
|---|---|---|
| 4 | 50 | ~200 |

---

## 🌍 Why This Matters

| Domain | Relevance |
|---|---|
| 🎓 Academic Integrity | Detecting AI-assisted submissions without watermarks |
| 📰 Misinformation Detection | Tracking synthetic propaganda to its source model |
| 🛡️ Cybersecurity | Investigating automated influence campaigns |
| ⚖️ AI Forensics | Legally-relevant attribution of generated content |
| 🤖 AI Safety Research | Understanding LLM behavioral patterns |

---

## 🤖 Models Being Studied


| Model | Provider | Platform |
|---|---|---|
| LLaMA 3.1 | Meta | Groq |
| GPT OSS | OpenAI | Groq |
| Qwen| AliBaba | Groq |
|  Gemma 4 26B | Google | AI Studio |

---

## ✨ Features — v1

### 🔎 Model Attribution
Paste any text → Clocked predicts which model wrote it with a confidence score and reasoning breakdown.

---

### 🧬 Stylometric Fingerprinting
Extracts hidden writing characteristics:
- Sentence rhythm and length distribution
- Lexical diversity and vocabulary richness
- Punctuation preferences
- Hedge language frequency
- Transition word dependency
- Structural consistency

---

### 💡 Explainable Predictions
Feature importance breakdown showing exactly which linguistic traits triggered the prediction.

#### Example Output
```
Detected Traits:
• High hedge-word frequency
• Formal explanatory structure
• Consistent paragraph symmetry
• Elevated transition usage

Most Likely Origin:
LLaMA 3.1 — 81%
```

---

### 🖥️ Interactive Dashboard
Clean Streamlit UI featuring:
- Confidence score bar chart per model
- Per-feature linguistic analysis
- Real-time text input and prediction

---

## 📦 Dataset Strategy

Since no benchmark dataset exists for this problem, the dataset is built manually by querying multiple LLMs using identical prompts.

### 🧾 Prompt Categories

| Category | Prompts | Example |
|---|---|---|
| 📘 Factual | 30 | "Explain how black holes form" |
| 💬 Argumentative | 30 | "Should social media be regulated?" |
| ✍️ Storytelling | 30 | "Write a story about a lighthouse keeper" |
| 💻 Coding Explanation | 30 | "Explain what a binary search tree is" |
| ❤️ Emotional | 30 | "How do I deal with losing a close friend?" |

**Total: 150 prompts × 5 models = 750+ labeled samples**

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---|---|
| Data Collection | Groq API |
| Data Collection | Google AI Studio |
| NLP & Features | spaCy, NLTK |
| Machine Learning | scikit-learn |
| Dashboard | Streamlit |
| Deployment | HuggingFace Spaces |
| Dataset Hosting | HuggingFace Datasets |

---

## 🗂️ Project Structure

```bash
Clocked/
│
├── 📁 data/
│   ├── prompts.csv              ← your 150 prompts
│   └── dataset.csv             ← collected + labeled responses
│
├── 📁 forensic_engine/
│   ├── collect.py              ← query models, save to CSV
│   ├── features.py             ← extract stylometric features
│   └── train.py                ← train & save classifier
│
├── app.py                      ← streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ System Workflow

```
Input Text
    ↓
🧹 Preprocessing
    ↓
🧬 Stylometric Feature Extraction
    ↓
🤖 Classifier (TF-IDF + SVM / Random Forest)
    ↓
📈 Explainability — Feature Importance
    ↓
🔍 Prediction + Confidence Score
```

---

## 📅 4-Week Roadmap

### ✅ Week 1 — Data Collection
> Target: 750+ labeled samples

- [ ] Sign up on Groq — get free API key
- [ ] Sign up on Google AI Studio — get free Gemma key
- [ ] Write 150 prompts across 5 categories
- [ ] Run `collect.py` — query all 5 models, save to CSV

---

### 🔬 Week 2 — Feature Engineering & Model
> Target: 75%+ accuracy

- [ ] Extract stylometric features with spaCy
- [ ] TF-IDF vectorization
- [ ] Train SVM + Random Forest classifiers
- [ ] Evaluate and pick best performing model

---

### 🖥️ Week 3 — Dashboard
> Target: Working UI

- [ ] Build Streamlit app — text input + prediction output
- [ ] Add confidence score bar chart
- [ ] Add feature importance breakdown
- [ ] Polish UI and test edge cases

---

### 🚀 Week 4 — Ship It
> Target: Live and public

- [ ] Deploy on HuggingFace Spaces
- [ ] Push dataset to HuggingFace Datasets
- [ ] Record demo video
- [ ] Add to portfolio and LinkedIn

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Clocked.git
```

### 2️⃣ Move Into the Project Folder
```bash
cd Clocked
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Collect the Dataset
```bash
python forensic_engine/collect.py
```

### 5️⃣ Train the Classifier
```bash
python forensic_engine/train.py
```

### 6️⃣ Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🧭 Version 2 — After v1 Ships

These features are planned for after the initial version is live:

- ⚔️ Adversarial robustness testing (paraphrasing attacks)
- 🕵️ Tamper and human-edit detection
- 🧠 Transformer fine-tuning for higher accuracy
- 📄 Forensic PDF report export
- 🌐 Browser extension for live detection
- 📝 Research paper submission

---

## 📌 Current Status

🚧 Project currently under active development.

Week 1 — dataset collection pipeline in progress.

---

## 👩‍💻 Author

**Disha S**

Built as part of independent exploration in:
- AI Forensics
- NLP
- Stylometry
- LLM Behavioral Analysis
**FROM SUMMER TASK OF OPENCODE.**

---

```
"Every model writes differently.
Clocked tries to uncover how."
```
