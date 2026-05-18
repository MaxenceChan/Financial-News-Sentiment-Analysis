# Financial News Sentiment Analysis

A comparative study of classical machine learning models and transformer-based architectures for sentiment classification on financial news headlines.

## 📋 Overview

This project benchmarks twelve different approaches to classify financial news sentences into three sentiment classes (**negative**, **neutral**, **positive**), comparing both performance and computational cost. The goal is to quantify the trade-off between simple lexical methods (TF-IDF) and modern contextual embeddings (BERT-based models).

## 🎯 Key Results

| Model | Macro-F1 | Training time | Cost (USD) |
|---|---|---|---|
| BernoulliNB | 0.48 | 0.002 s | $3 × 10⁻⁸ |
| Logistic Regression (TF-IDF + SVD) | **0.64** | 0.07 s | $10⁻⁶ |
| Sentence-BERT + LogReg | 0.73 | 12 s | $1.7 × 10⁻³ |
| **DistilBERT fine-tuned** | **0.82** | 176 s | $2.5 × 10⁻² |

DistilBERT fine-tuned achieves the best performance, but Logistic Regression on TF-IDF features captures ~78% of its macro-F1 at a fraction of the cost.

## 📚 Dataset

This project uses the [**Financial PhraseBank**](https://huggingface.co/datasets/takala/financial_phrasebank) dataset, available on Hugging Face. It consists of ~4,840 English sentences from financial news, manually annotated with sentiment labels (negative / neutral / positive) by 16 annotators with financial background.

The dataset is class-imbalanced:
- **Neutral**: ~59%
- **Positive**: ~28%
- **Negative**: ~13%

## 🗂️ Project Structure

```
Financial-News-Sentiment-Analysis/
├── data/                       # Raw and processed datasets
├── Code.ipynb                  # Main analysis notebook
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Classical ML**: scikit-learn, scipy
- **NLP**: spaCy, NLTK
- **Transformers**: Hugging Face `transformers`, `sentence-transformers`, PyTorch
- **Analysis**: pandas, numpy, matplotlib

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Financial-News-Sentiment-Analysis.git
cd Financial-News-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

The dataset is loaded directly from Hugging Face — no manual download required:

```python
from datasets import load_dataset
ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
```

## 📊 Methodology

The project follows a three-part pipeline:

**1. Preprocessing**
Text cleaning, tokenization, lemmatization, and stopword removal using spaCy.

**2. Classical models (TF-IDF + SVD)**
TF-IDF vectorization followed by dimensionality reduction via Truncated SVD (`n_components=500`, chosen empirically by tracking macro-F1). Models compared: Logistic Regression, Gaussian/Bernoulli Naive Bayes, Decision Tree, Random Forest.

**3. Transformer-based models**
- **Sentence-BERT** (`all-MiniLM-L6-v2`) used as a frozen encoder + Logistic Regression head.
- **DistilBERT** (`distilbert-base-uncased`) fine-tuned end-to-end on the task.

All models are evaluated on the same train/test split using:
- **Macro-F1** as the primary metric (chosen over accuracy due to class imbalance)
- **MSE** on ordinal labels (to capture severity of misclassifications)
- **Total time** (training + inference) and **estimated computational cost**

## 📈 Why Macro-F1 instead of Accuracy?

The Financial PhraseBank is imbalanced (~60% neutral, ~28% positive, ~12% negative). Accuracy would be mechanically inflated by performance on the majority class and mask weaknesses on the minority ones. Macro-F1 weights each class equally and reflects the operational usefulness of the model — crucial since detecting negative financial news is often more valuable than classifying yet another neutral headline.

## 💡 Key Takeaways

- **Transformers dominate raw performance**, with fine-tuned DistilBERT reaching 0.82 macro-F1.
- **Logistic Regression on TF-IDF + SVD is a remarkably strong baseline**, achieving 0.64 macro-F1 in 70 ms — roughly 700,000× cheaper than DistilBERT for ~78% of its performance.
- **Naive Bayes, Decision Trees, and Random Forests underperform** on this task (0.39–0.48 macro-F1), suggesting that the bottleneck lies in the model class itself rather than in feature engineering.
- **Choice of vectorization matters more than choice of classifier**: moving from TF-IDF to contextual embeddings yields a larger gain than tuning any individual model.

## 📄 License

This project is provided for educational purposes.