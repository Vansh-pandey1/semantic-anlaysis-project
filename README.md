# 🧠 Semantic Analysis Project

> A machine learning–powered sentiment classifier that detects whether a given text statement is **Positive**, **Negative**, **Neutral**, or **Irrelevant** — served through an interactive Streamlit web app.

**Author:** Vansh Pandey

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Limitations](#limitations)
- [Contributing](#contributing)

---

## Overview

This project performs **semantic (sentiment) analysis** on free-form text input. It uses a pre-trained machine learning model backed by TF-IDF vectorization and NLTK-based text preprocessing to classify statements into one of four sentiment categories:

| Label | Meaning |
|-------|---------|
| 😡 Negative | The statement carries a negative/critical tone |
| 😐 Neutral | The statement is factual or emotionally flat |
| 🤗 Positive | The statement carries a positive/optimistic tone |
| 🥱 Irrelevant | The statement does not belong to any meaningful category |

The full model training and exploratory analysis lives in `semantic_analysis.ipynb`, while `app.py` serves as the Streamlit front-end for real-time predictions.

---

## Features

- **Real-time sentiment prediction** from user-typed input
- **NLP preprocessing pipeline**: tokenization, stopword removal, punctuation filtering, and Porter stemming
- **TF-IDF vectorization** for numeric feature extraction
- **Interactive Streamlit UI** — no coding required to use
- **Jupyter notebook** with complete EDA, model training, and evaluation
- Visualizations using `matplotlib`, `seaborn`, and `wordcloud`

---

## Project Structure

```
semantic-analysis-project/
│
├── dataset/                    # Raw training/evaluation data
│
├── semantic_analysis.ipynb     # Full EDA, preprocessing, model training & evaluation
│
├── app.py                      # Streamlit web application (front-end + inference)
├── main.py                     # Entry-point placeholder
│
├── vectorizer.pkl              # Saved TF-IDF vectorizer (fitted on training data)
├── model.pkl                   # Saved trained classifier (not tracked by git — generate via notebook)
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata (uv-compatible)
├── uv.lock                     # Locked dependency versions
├── .python-version             # Python version pin
├── .gitignore
└── README.md
```

---

## How It Works

The prediction pipeline follows these steps:

```
User Input (raw text)
        │
        ▼
  Lowercase conversion
        │
        ▼
  Word tokenization (NLTK)
        │
        ▼
  Remove non-alphanumeric tokens
        │
        ▼
  Remove stopwords & punctuation
        │
        ▼
  Porter Stemming
        │
        ▼
  TF-IDF Vectorization (vectorizer.pkl)
        │
        ▼
  ML Model Inference (model.pkl)
        │
        ▼
  Sentiment Label (Positive / Negative / Neutral / Irrelevant)
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Web Framework | Streamlit |
| NLP | NLTK (tokenization, stopwords, stemming) |
| ML | scikit-learn (TF-IDF, classifier) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, wordcloud |
| Notebook | Jupyter / ipykernel |
| Package Manager | uv |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vansh-pandey1/semantic-anlaysis-project.git
cd semantic-anlaysis-project
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using [uv](https://github.com/astral-sh/uv) (faster):

```bash
uv sync
```

### 4. Download NLTK data

The app downloads required NLTK corpora automatically on first run, but you can also do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

## Running the App

> **Important:** Make sure `model.pkl` exists in the project root. If it doesn't, run through the `semantic_analysis.ipynb` notebook first to train and export the model.

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**Usage:**
1. Type or paste a statement into the text area.
2. Click the **Predict** button.
3. The app will display the predicted sentiment label.

---

## Dataset

The training data is located in the `dataset/` folder. It contains labeled text samples used to train and evaluate the classifier. The notebook (`semantic_analysis.ipynb`) covers:

- Data loading and inspection
- Class distribution analysis
- Text cleaning and preprocessing
- Word cloud and frequency visualizations
- Model training and accuracy metrics

---

## Model Details

- **Vectorizer:** TF-IDF (`TfidfVectorizer` from scikit-learn), fitted on the training corpus and saved as `vectorizer.pkl`
- **Classifier:** A supervised ML model (trained and serialized as `model.pkl` via the notebook)
- **Labels:** 4-class classification — `1` (Negative), `2` (Neutral), `3` (Positive), `0/other` (Irrelevant)

To retrain the model, open and run all cells in `semantic_analysis.ipynb`. The notebook will regenerate both `vectorizer.pkl` and `model.pkl`.

---

## Limitations

- The model can sometimes **confuse Neutral and Irrelevant** statements — this is a known limitation of the current training data and model.
- Performance may vary on very short or highly ambiguous inputs.
- The model is trained on a specific domain/corpus; out-of-domain text may produce unreliable results.

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## License

This project is open-source. Feel free to use, modify, and distribute with attribution.

---

*Made with 🤗 by [Vansh Pandey](https://github.com/Vansh-pandey1)*
