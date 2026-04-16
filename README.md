# Financial News Sentiment Analysis

A multi-model NLP project for sentiment classification of financial news, progressing from classical machine learning to fine-tuned transformer models.

## Overview

This project classifies financial news sentences into three sentiment categories — **negative**, **neutral**, and **positive** — using the [auditor_sentiment](https://huggingface.co/datasets/FinanceInc/auditor_sentiment) dataset from Hugging Face. The goal is to extract actionable sentiment signals from financial text, with applications in market analysis and risk assessment.

## Dataset

- **Source:** Hugging Face — `FinanceInc/auditor_sentiment`
- **Task:** Multi-class text classification (3 classes: negative / neutral / positive)
- **Split:** 80% training / 20% test (stratified)
- Preprocessing includes duplicate removal, class distribution analysis, and profiling via `ydata-profiling`

## Methodology

The project is structured in 4 progressive classifier generations:

### Part 1 — Data Preparation
- Text cleaning: lowercasing, special character removal, stopword filtering (NLTK)
- Lemmatization with **spaCy** (`en_core_web_sm`)
- Vectorization with **TF-IDF**
- Dimensionality reduction with **TruncatedSVD** (170 components)

### Part 2 — Classifier 1.0 (TF-IDF baseline)
- Logistic Regression
- Decision Tree
- Random Forest

### Part 3 — Classifier 2.0 (hyperparameter tuning)
- Logistic Regression (grid search)
- Random Forest (grid search over `n_estimators`, `max_depth`, `max_features`)
- Decision Tree (grid search over `max_depth`, `criterion`, `min_samples_*`)
- Gaussian Naive Bayes
- Bernoulli Naive Bayes (on binarized TF-IDF)

### Part 4 — Classifier 3.0 (Sentence Embeddings)
- **Sentence-BERT** (`all-MiniLM-L6-v2`) for dense text embeddings
- Logistic Regression on top of SBERT vectors

### Part 5 — Classifier 4.0 (Fine-Tuned Transformer)
- **DistilBERT** (`distilbert-base-uncased`) fine-tuned for sequence classification
- 3-class output head, trained with HuggingFace `Trainer` API
- Dynamic padding, `learning_rate=2e-5`, `weight_decay=0.01`

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3D7EAA?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

**Key libraries:** `transformers`, `sentence-transformers`, `datasets`, `scikit-learn`, `spacy`, `nltk`, `pandas`, `ydata-profiling`

## Project Structure

```
Financial-News-Sentiment-Analysis/
├── Code.ipynb       # Main notebook (all 5 parts)
├── train.csv        # Training set (80%)
└── test.csv         # Test set (20%)
```

## Getting Started

```bash
git clone https://github.com/MaxenceChan/Financial-News-Sentiment-Analysis.git
cd Financial-News-Sentiment-Analysis

pip install pandas scikit-learn nltk spacy ydata-profiling \
            sentence-transformers transformers datasets evaluate \
            accelerate torch

python -m spacy download en_core_web_sm
jupyter notebook Code.ipynb
```

## Author

**Maxence Chan** — Data Scientist | Finance & Risk  
[LinkedIn](https://linkedin.com/in/maxence-chan) · [GitHub](https://github.com/MaxenceChan)
