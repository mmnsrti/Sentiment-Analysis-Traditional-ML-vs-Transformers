# 🎬 Sentiment Analysis: Traditional ML vs Transformers

An end-to-end **IMDb sentiment classification** project comparing a strong traditional NLP pipeline — **TF-IDF + PyTorch Logistic Regression** — against a fine-tuned **DistilBERT Transformer**.

The project covers the complete machine-learning workflow:

**Data validation → EDA → Traditional NLP baseline → Error analysis → Controlled experiments → Transformer fine-tuning → Final test evaluation → Model comparison → Inference**

> **Key result:** In this experimental setup, the optimized TF-IDF model slightly outperformed DistilBERT on the official test set while being significantly cheaper to run.

---

## 🏆 Final Results

All final metrics were measured on the same **25,000-review IMDb test set**.

| Model                       |   Accuracy |  Precision |     Recall |         F1 |
| --------------------------- | ---------: | ---------: | ---------: | ---------: |
| TF-IDF Unigram Baseline     |     85.75% |     85.63% |     85.91% |     85.77% |
| **TF-IDF Unigram + Bigram** | **87.62%** |     86.88% | **88.63%** | **87.75%** |
| DistilBERT                  |     87.36% | **88.08%** |     86.40% |     87.23% |

### Winner

**TF-IDF + Logistic Regression with unigram + bigram features**

```text
Test Accuracy : 0.8762
Test F1       : 0.8775
Macro F1      : 0.8762
```

DistilBERT achieved:

```text
Test Accuracy : 0.8736
Test F1       : 0.8723
Test Loss     : 0.3114
```

The difference is small, but important:

```text
DistilBERT Δ F1       = -0.0051
DistilBERT Δ Accuracy = -0.0027
```

This project therefore demonstrates an important engineering lesson:

> **A more complex model is not automatically the better model for every dataset, constraint, or experimental setup.**

---

## 🎯 Project Goal

The goal of this project was not simply to maximize IMDb accuracy.

The project was designed to understand and compare **two generations of NLP modeling**:

### Traditional NLP

```text
Raw Review
    ↓
Text Cleaning
    ↓
TF-IDF
    ↓
Sparse Feature Vector
    ↓
Logistic Regression
    ↓
Sentiment
```

### Transformer NLP

```text
Raw Review
    ↓
Tokenizer
    ↓
Input IDs + Attention Mask
    ↓
DistilBERT
    ↓
Contextual Representation
    ↓
Classification Head
    ↓
Sentiment
```

Both approaches were evaluated using the same cleaned train, validation, and test data.

---

# 📊 Dataset

The project uses the **IMDb binary movie-review sentiment dataset**.

Labels:

```text
0 → Negative
1 → Positive
```

After data-quality and leakage checks, the final splits contain:

| Split      | Reviews |
| ---------- | ------: |
| Train      |  19,823 |
| Validation |   4,956 |
| Test       |  25,000 |

The training set is almost perfectly balanced:

```text
Negative : 49.79%
Positive : 50.21%
```

---

## 🔍 Exploratory Data Analysis

Several quality checks were performed before modeling.

### Data quality

```text
Missing texts              : 0
Empty texts                : 0
Duplicate train rows       : 0
Train/Validation overlap   : 0
```

### Review length

```text
Average words : 233
Median words  : 174
95th percentile : 595
99th percentile : ~909
Maximum       : 2,470
```

Approximately **58.77%** of training reviews originally contained HTML-style formatting such as `<br />`.

These findings later become particularly relevant when comparing TF-IDF with a Transformer that uses a fixed maximum sequence length.

---

# 🧹 Text Processing

The traditional pipeline applies lightweight cleaning intended to remove formatting without aggressively modifying linguistic information.

Examples include:

```text
HTML entity decoding
<br /> removal
HTML tag removal
Whitespace normalization
```

No stemming or aggressive stop-word removal is used.

For DistilBERT, the pretrained tokenizer handles the model-specific tokenization process.

---

# 🧠 Model 1 — TF-IDF + PyTorch Logistic Regression

The first real NLP model converts each review into a sparse TF-IDF vector.

Initial configuration:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    max_features=20_000,
    sublinear_tf=True,
    dtype=np.float32,
)
```

The resulting training matrix has:

```text
Shape   : (19,823 × 20,000)
Density : ~0.65%
```

Because the matrix is extremely sparse, it is kept in sparse format and converted to dense tensors **one batch at a time** instead of densifying the entire dataset.

---

## PyTorch Logistic Regression

The classifier is intentionally simple:

```text
20,000 TF-IDF features
        ↓
nn.Linear(20,000, 1)
        ↓
Logit
        ↓
Sigmoid
        ↓
P(positive)
```

Number of trainable parameters:

```text
20,001
```

Training configuration:

```text
Loss          : BCEWithLogitsLoss
Optimizer     : AdamW
Learning rate : 1e-3
Weight decay  : 1e-4
Batch size    : 128
Epochs        : 8
```

Best unigram validation F1:

```text
0.8687
```

Final unigram test F1:

```text
0.8577
```

---

# 🔎 Traditional Model Error Analysis

The baseline was not treated as a black box.

The project includes detailed analysis of:

* False positives
* False negatives
* High-confidence mistakes
* Near-boundary predictions
* Errors by review length
* Global feature importance
* Per-review feature contribution

The baseline produced:

```text
True Negatives  : 10,698
False Positives : 1,802
False Negatives : 1,761
True Positives  : 10,739
```

---

## Feature Interpretability

Because Logistic Regression operates directly on TF-IDF features, individual learned weights can be inspected.

Some strongly positive unigram features included:

```text
great
excellent
wonderful
perfect
best
amazing
favorite
loved
fantastic
brilliant
```

Strong negative features included:

```text
worst
bad
awful
waste
worse
poor
boring
terrible
horrible
stupid
```

A prediction can also be decomposed using:

```text
feature contribution
=
TF-IDF value × learned model weight
```

This provides a level of interpretability that is considerably harder to obtain from a Transformer model.

---

# 🧪 Controlled TF-IDF Experiment

The next question was:

> Can a simple traditional model improve if it captures short phrases rather than only individual words?

Two controlled experiments were run while keeping all other important settings unchanged.

| Experiment                    | Features | Validation Accuracy | Validation F1 |
| ----------------------------- | -------: | ------------------: | ------------: |
| Unigram `(1, 1)`              |   20,000 |              86.64% |        86.92% |
| **Unigram + Bigram `(1, 2)`** |   20,000 |          **88.03%** |    **88.29%** |

Improvement:

```text
Δ Validation F1       : +0.0137
Δ Validation Accuracy : +0.0139
```

The selected traditional model therefore became:

```text
TF-IDF
ngram_range=(1, 2)
max_features=20,000
+
PyTorch Logistic Regression
```

Its final test performance was:

```text
Accuracy  : 87.62%
Precision : 86.88%
Recall    : 88.63%
F1        : 87.75%
```

---

# 🤗 Model 2 — DistilBERT

The second modeling approach uses:

```text
distilbert-base-uncased
```

Instead of manually constructing statistical features, the Transformer receives sequences of tokens.

The project explicitly explores:

```text
Subword tokenization
Token IDs
Special tokens
Padding
Dynamic padding
Attention masks
Truncation
Contextual representations
Softmax classification
Transfer learning
Fine-tuning
```

---

## Transformer Input

A review is transformed roughly as:

```text
"I did not like this movie"

        ↓

Tokenizer

        ↓

[CLS] i did not like this movie [SEP]

        ↓

Input IDs
+
Attention Mask

        ↓

DistilBERT
```

Unlike TF-IDF, Transformer token representations depend on their surrounding context.

---

# ⚙️ DistilBERT Fine-Tuning

Configuration used in this project:

```text
Model          : distilbert-base-uncased
Classes        : 2
Max length     : 128 tokens

Train batch    : 16
Validation     : 32

Optimizer      : AdamW
Learning rate  : 2e-5
Gradient clip  : 1.0

Epochs         : 2
```

Best validation result:

```text
Validation Accuracy : 0.8745
Validation F1       : 0.8737
```

Training time for two epochs in the recorded GPU run:

```text
~242 seconds
```

Final test performance:

```text
Accuracy  : 87.36%
Precision : 88.08%
Recall    : 86.40%
F1        : 87.23%
```

---

# 🤔 Why Did TF-IDF Beat DistilBERT?

This is one of the most interesting outcomes of the project.

DistilBERT has dramatically more expressive modeling capacity and understands context better than TF-IDF.

Yet the optimized traditional model achieved slightly higher test F1.

Several factors may contribute.

### 1. IMDb contains strong lexical sentiment signals

Words and short phrases such as:

```text
excellent
terrible
waste
not good
very bad
highly recommend
```

are extremely informative for this task.

TF-IDF with bigrams can capture many of these signals very efficiently.

### 2. The Transformer was limited to 128 tokens

The training review has a median length of **174 words**, while the 95th percentile is approximately **595 words**.

The DistilBERT configuration therefore truncates a meaningful portion of many reviews.

This is a plausible contributor to the result, although the project does not experimentally prove that truncation is the cause of the performance difference.

### 3. DistilBERT received minimal tuning

The Transformer experiment intentionally remained a baseline:

```text
2 epochs
2e-5 learning rate
128-token context
```

There was no extensive Transformer hyperparameter search.

### 4. Model complexity has a cost

The traditional classifier uses only:

```text
20,001 trainable parameters
```

after TF-IDF feature extraction.

DistilBERT contains millions of parameters and requires substantially more compute.

For this task, that extra complexity did not produce a higher final F1 under the tested configuration.

---

# ⚖️ Traditional ML vs Transformer

| Property                 | TF-IDF + Logistic Regression | DistilBERT                       |
| ------------------------ | ---------------------------- | -------------------------------- |
| Representation           | Sparse statistical features  | Contextual token representations |
| Context understanding    | Limited                      | Strong                           |
| Training cost            | Low                          | High                             |
| Inference cost           | Low                          | Higher                           |
| Interpretability         | High                         | Lower                            |
| Deployment complexity    | Low                          | Higher                           |
| Final F1 in this project | **87.75%**                   | 87.23%                           |

The central engineering takeaway is:

> **Choose models based on measured performance, cost, latency, interpretability, and deployment constraints — not only model sophistication.**

---

# 📁 Project Structure

```text
.
├── data/
│   └── processed/
│       └── imdb_clean_splits/
│
├── models/
│   ├── best_traditional/
│   │   ├── tfidf_logistic_regression.pt
│   │   └── tfidf_vectorizer.joblib
│   │
│   └── distilbert_imdb/
│       ├── model files
│       ├── config
│       └── tokenizer files
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tfidf_pytorch_baseline.ipynb
│   ├── 03_tfidf_test_evaluation.ipynb
│   ├── 04_tfidf_experiments.ipynb
│   ├── 05_best_traditional_test.ipynb
│   ├── 06_distilbert_baseline.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_inference_demo.ipynb
│
├── reports/
│   ├── baseline/
│   ├── best_traditional/
│   ├── distilbert/
│   ├── tfidf_experiments/
│   └── final_comparison/
│
└── README.md
```

---

# 📓 Notebook Guide

### `01_eda.ipynb`

Exploratory data analysis and data-quality validation.

Covers class balance, missing values, duplicate checks, review-length statistics, and HTML content.

### `02_tfidf_pytorch_baseline.ipynb`

Builds the first sentiment classifier using TF-IDF and a PyTorch Logistic Regression model.

### `03_tfidf_test_evaluation.ipynb`

Evaluates the baseline on the official test set and performs detailed error analysis and feature interpretation.

### `04_tfidf_experiments.ipynb`

Runs a controlled unigram vs unigram+bigram experiment and selects the best traditional model.

### `05_best_traditional_test.ipynb`

Performs the final test evaluation of the selected traditional model.

### `06_distilbert_baseline.ipynb`

Introduces Transformer tokenization concepts and fine-tunes DistilBERT on IMDb.

### `07_model_comparison.ipynb`

Compares the final traditional and Transformer models.

### `08_inference_demo.ipynb`

Loads the saved Transformer and performs sentiment prediction on new, unseen reviews.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/mmnsrti/Sentiment-Analysis-Traditional-ML-vs-Transformers
cd Sentiment-Analysis-Traditional-ML-vs-Transformers
```

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy scipy matplotlib joblib jupyter
```

Then start Jupyter:

```bash
jupyter lab
```

---

# ▶️ Running the Project

The notebooks are designed to be followed in numerical order:

```text
01 → EDA
02 → TF-IDF baseline
03 → Baseline test + error analysis
04 → Controlled experiments
05 → Best traditional model test
06 → DistilBERT fine-tuning
07 → Final comparison
08 → Inference demo
```

The notebooks expect the processed Hugging Face `DatasetDict` at:

```text
data/processed/imdb_clean_splits/
```

---

# 🔮 Inference

The final DistilBERT model and tokenizer are loaded from:

```text
models/distilbert_imdb/
```

After running the inference notebook:

```python
result = predict_sentiment(
    "This movie was absolutely fantastic. "
    "I loved every minute of it."
)

print(result)
```

Example output from the trained model:

```text
label: pos
negative_probability: 0.0171
positive_probability: 0.9829
```

Batch inference is also supported.

```python
reviews = [
    "Absolutely brilliant movie. I loved it.",
    "Terrible film. A complete waste of time.",
    "It started badly but became really enjoyable.",
]

predict_sentiment_batch(reviews)
```

---

# 📈 Generated Reports

The project saves intermediate and final analysis rather than keeping important results only inside notebook memory.

Examples include:

```text
classification_report.csv
confusion_matrix.csv
test_metrics.csv
test_predictions.csv
false_positives.csv
false_negatives.csv
feature_weights.csv
top_sentiment_features.csv
error_by_review_length.csv
experiment_comparison.csv
training_histories.csv
model_metrics_comparison.csv
model_tradeoffs.csv
```

This makes the experiments easier to inspect and reproduce.

---

# ✅ Evaluation Protocol

Several rules were followed to reduce leakage and overly optimistic evaluation:

1. Model parameters are learned from the training split.
2. Hyperparameter/model decisions are based on validation performance.
3. The final test set is used for final evaluation rather than model fitting.
4. The traditional and Transformer approaches use the same dataset splits.
5. Duplicate/overlap checks are performed before modeling.
6. Controlled TF-IDF experiments keep unrelated training settings fixed.
7. Random seeds are fixed where applicable.

---

# 🧪 Possible Future Work

The current project is intentionally a focused comparison rather than a large hyperparameter search.

Natural extensions include:

* Increase DistilBERT context length to 256 or 512 tokens
* Experiment with long-review chunking instead of simple truncation
* Add learning-rate scheduling and warmup
* Compare 2 vs 3 Transformer fine-tuning epochs
* Perform Transformer-specific error analysis
* Compare predictions where TF-IDF and DistilBERT disagree
* Evaluate probability calibration
* Try additional pretrained Transformer architectures
* Export the final classifier behind a REST API
* Build a lightweight web interface for interactive predictions
* Benchmark CPU inference and deployment memory explicitly

---

# 💡 What This Project Demonstrates

This repository is more than a sentiment-classification notebook.

It demonstrates a complete modeling workflow and the progression from classical NLP to modern Transformer-based NLP:

```text
Data quality
    ↓
Exploratory analysis
    ↓
Simple baseline
    ↓
Sparse NLP representation
    ↓
PyTorch training
    ↓
Evaluation
    ↓
Error analysis
    ↓
Interpretability
    ↓
Controlled experimentation
    ↓
Transfer learning
    ↓
Transformer fine-tuning
    ↓
Final model selection
    ↓
Production-style inference
```

Most importantly, it demonstrates that model selection should be driven by **evidence rather than complexity**.

In this IMDb experiment, the lightweight and interpretable **TF-IDF + Logistic Regression** pipeline remained the strongest overall solution.

---

## Technologies

**Python · PyTorch · scikit-learn · Hugging Face Transformers · Hugging Face Datasets · pandas · NumPy · SciPy · Matplotlib · Jupyter**

---

## Acknowledgements

This project uses the IMDb movie-review sentiment dataset and builds on the open-source ecosystems provided by PyTorch, scikit-learn, and Hugging Face.

---

⭐ If you found this project useful, consider starring the repository.
