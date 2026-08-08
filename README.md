# Sentiment Analysis: Traditional ML vs Transformers

An end-to-end IMDb sentiment-classification project comparing a strong **TF-IDF + PyTorch Logistic Regression** pipeline with a fine-tuned **DistilBERT** model.

The project covers data cleaning, leakage checks, EDA, sparse NLP modeling, controlled experiments, Transformer fine-tuning, final evaluation, model comparison, and inference.

## Key result

On the 25,000-review IMDb test set, the optimized traditional model slightly outperformed the DistilBERT baseline used in this project.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF Unigram Baseline | 85.75% | 85.63% | 85.91% | 85.77% |
| **TF-IDF Unigram + Bigram** | **87.62%** | 86.88% | **88.63%** | **87.75%** |
| DistilBERT | 87.36% | **88.08%** | 86.40% | 87.23% |

**Best model:** TF-IDF + Logistic Regression with unigram + bigram features.

The main engineering takeaway is simple:

> More complex does not automatically mean better. Model choice should be based on measured quality, cost, latency, interpretability, and deployment constraints.

## What is inside

- leakage-aware IMDb data preparation
- exploratory data analysis
- TF-IDF + PyTorch Logistic Regression baseline
- test-set error analysis and feature interpretation
- controlled unigram vs unigram+bigram experiment
- final traditional-model evaluation
- DistilBERT tokenization and fine-tuning
- final model comparison
- single-text and batch inference demos

## Pipeline

```text
IMDb
  ↓
Data cleaning & decontamination
  ↓
Train / Validation / Test
  ↓
EDA
  ↓
├── TF-IDF → Logistic Regression → Error Analysis → Bigram Experiment
│
└── Tokenizer → DistilBERT → Fine-tuning
  ↓
Final Comparison
  ↓
Inference Demo
```

## Repository guide

```text
notebooks/
├── 01_eda.ipynb
├── 02_tfidf_pytorch_baseline.ipynb
├── 03_tfidf_test_evaluation.ipynb
├── 04_tfidf_experiments.ipynb
├── 05_best_traditional_test.ipynb
├── 06_distilbert_baseline.ipynb
├── 07_model_comparison.ipynb
└── 08_inference_demo.ipynb
```

For methodology, experiments, metrics, limitations, reproducibility notes, and a notebook-by-notebook walkthrough, see:

**[Full project documentation →](docs/PROJECT_DETAILS.md)**

## Quick start

Install the main dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy scipy matplotlib joblib jupyter
```

Then open Jupyter and run the notebooks in numerical order:

```bash
jupyter lab
```

The cleaned dataset is expected at:

```text
data/processed/imdb_clean_splits/
```

Saved models are expected under:

```text
models/
├── best_traditional/
└── distilbert_imdb/
```

## Example inference

The final inference notebook loads the saved DistilBERT tokenizer and model:

```python
result = predict_sentiment(
    "This movie was absolutely fantastic. I loved every minute of it."
)
```

Recorded output:

```text
Prediction: pos
Negative probability: 0.0171
Positive probability: 0.9829
```

## Tech stack

`Python` · `PyTorch` · `scikit-learn` · `Hugging Face Transformers` · `Hugging Face Datasets` · `pandas` · `NumPy` · `SciPy` · `Matplotlib` · `Jupyter`

---

**Evaluation note:** the baseline test set was analyzed before the later bigram experiment. The unigram-vs-bigram choice itself was made on validation performance, but the final test set was not a perfectly untouched holdout throughout the entire project workflow. See the detailed documentation for the full caveat.
