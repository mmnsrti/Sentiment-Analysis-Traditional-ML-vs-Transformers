# Project Details

## Sentiment Analysis: Traditional ML vs Transformers

This document contains the full methodology and results for the IMDb sentiment-classification project.

The project compares two modeling families:

1. **Traditional NLP:** TF-IDF features + PyTorch Logistic Regression
2. **Transformer NLP:** fine-tuned DistilBERT

The goal was not only to obtain a sentiment score, but to build and understand a complete machine-learning workflow:

```text
Data quality
→ EDA
→ Baseline
→ Evaluation
→ Error analysis
→ Controlled experiment
→ Better traditional model
→ Transformer fine-tuning
→ Final comparison
→ Inference
```

---

# 1. Problem definition

The task is binary sentiment classification on IMDb movie reviews.

Labels:

```text
0 = neg
1 = pos
```

For each review, the model predicts whether the overall sentiment is negative or positive.

The project intentionally compares two very different representations of language:

### Traditional approach

```text
Review
→ mild text cleanup
→ TF-IDF
→ sparse feature vector
→ Logistic Regression
→ sentiment
```

### Transformer approach

```text
Review
→ pretrained tokenizer
→ token IDs + attention mask
→ DistilBERT
→ classification head
→ sentiment
```

---

# 2. Dataset and split construction

The source dataset is the Stanford IMDb sentiment dataset loaded through Hugging Face Datasets.

The final cleaned split sizes used throughout the modeling notebooks are:

| Split | Rows |
|---|---:|
| Train | 19,823 |
| Validation | 4,956 |
| Test | 25,000 |

The official IMDb test split is preserved as the test set.

The training pool is cleaned and then split into train and validation using a stratified 80/20 split with a fixed random seed.

Final label mapping:

```text
0 → neg
1 → pos
```

---

# 3. Data cleaning and leakage prevention

The project contains a dedicated cleaning workflow before modeling.

A normalized comparison key is created only for duplicate detection. The original review text is preserved for training.

Normalization for duplicate detection includes:

- Unicode NFKC normalization
- HTML entity decoding
- replacement of IMDb `<br>` tags
- removal of remaining HTML tags
- whitespace normalization
- case folding

The cleaning workflow then:

1. detects duplicate normalized texts with conflicting labels
2. removes all rows belonging to conflicting duplicate groups
3. removes internal duplicate reviews from the training pool
4. removes training rows that also occur in the official test set
5. creates a new stratified train/validation split
6. verifies that duplicate and overlap checks pass

Final checks assert:

```text
train internal duplicates      = 0
validation internal duplicates = 0
train-validation overlap       = 0
train-test overlap             = 0
validation-test overlap        = 0
```

This cleaned dataset is saved as:

```text
data/processed/imdb_clean_splits/
```

---

# 4. Exploratory data analysis

The EDA notebook examines class balance, text quality, duplication, review length, and HTML content.

## 4.1 Class balance

Training-set distribution:

| Label | Count | Percentage |
|---|---:|---:|
| Negative | 9,869 | 49.79% |
| Positive | 9,954 | 50.21% |

The training set is therefore close to perfectly balanced.

## 4.2 Missing and duplicate text

The EDA reports:

```text
Missing texts: 0
Empty texts:   0
Rows involved in duplicate groups: 0
Extra duplicated rows: 0
Identical train/validation reviews: 0
```

These values refer to the already-cleaned dataset used by the notebook.

## 4.3 Review length

Training review statistics:

| Statistic | Word count |
|---|---:|
| Mean | 233.40 |
| Median | 174 |
| 75th percentile | 283 |
| 90th percentile | 455 |
| 95th percentile | 595 |
| 99th percentile | 908.78 |
| Maximum | 2,470 |

The distribution is strongly right-skewed: most reviews are much shorter than the longest examples, but a meaningful tail contains very long reviews.

This becomes important later because the Transformer baseline uses a maximum sequence length of 128 tokens.

## 4.4 HTML content

Reviews containing HTML-like tags:

```text
11,650
```

Percentage of training reviews containing HTML:

```text
58.77%
```

For the traditional model, mild formatting cleanup removes these artifacts without aggressive linguistic preprocessing.

---

# 5. Evaluation protocol and caveat

The intended protocol is:

```text
Train      → fit model parameters
Validation → compare configurations / select checkpoints
Test       → report final generalization
```

The unigram-vs-bigram experiment itself follows this idea: both alternatives are compared on validation performance and the winner is then evaluated on the official test set.

However, there is an important caveat.

Notebook `03_tfidf_test_evaluation.ipynb` performs detailed error analysis on the official test set before the later bigram experiment. The notebook explicitly examines false positives, false negatives, high-confidence mistakes, review length, and limitations of unigram features. It also identifies bigrams as a possible next experiment.

Therefore, although the **actual unigram-vs-bigram selection was based on validation metrics**, the final test set was not a perfectly untouched holdout throughout the entire project-development process.

For a stricter research protocol, the recommended redesign would be:

```text
Train
  ↓
Validation
  ├── error analysis
  ├── feature experiments
  └── model selection
  ↓
Final locked model
  ↓
Test exactly once
```

The current results remain useful as an educational and engineering comparison, but this caveat should be kept in mind when interpreting small final differences.

---

# 6. Traditional NLP baseline

## 6.1 Text cleanup

The traditional pipeline applies mild formatting cleanup:

- HTML entity decoding
- `<br>` replacement
- remaining HTML tag removal
- repeated-whitespace normalization

The goal is to remove formatting artifacts while preserving the linguistic content of each review.

No aggressive stemming or stop-word removal is applied.

---

# 7. TF-IDF representation

The original baseline vectorizer uses:

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

The fitted training matrix has:

```text
Shape:   (19,823, 20,000)
NNZ:     2,580,136
Density: 0.650794%
```

Because the representation is sparse, converting the entire dataset to a dense matrix would waste substantial memory.

The project therefore keeps the full dataset in SciPy sparse format and converts only the current mini-batch to dense NumPy/PyTorch tensors.

This allows a standard PyTorch linear classifier to be trained without densifying the full training matrix.

---

# 8. PyTorch Logistic Regression

The model architecture is intentionally minimal:

```text
20,000 TF-IDF features
        ↓
nn.Linear(20,000, 1)
        ↓
single logit
        ↓
sigmoid probability
```

Trainable parameters:

```text
20,001
```

The model outputs one logit per review.

During inference:

```text
sigmoid(logit) = P(positive)
```

Prediction rule:

```text
P(positive) >= 0.5 → positive
P(positive) <  0.5 → negative
```

## 8.1 Training configuration

```text
Loss:          BCEWithLogitsLoss
Optimizer:     AdamW
Learning rate: 1e-3
Weight decay:  1e-4
Batch size:    128
Epochs:        8
Random seed:   42
```

The best model state is selected by validation F1.

## 8.2 Baseline validation result

Best validation F1:

```text
0.8687
```

The best value occurs at epoch 8 in the baseline run.

---

# 9. Baseline test evaluation

The saved unigram TF-IDF model is evaluated on all 25,000 official test reviews.

Final baseline metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8575 |
| Precision | 0.8563 |
| Recall | 0.8591 |
| F1 | 0.8577 |
| Macro F1 | 0.8575 |

Recorded inference throughput in the notebook:

```text
~19,250 reviews/second
```

This throughput reflects the recorded execution environment and should not be treated as a hardware-independent benchmark.

---

# 10. Error analysis and interpretability

The baseline test notebook goes beyond aggregate metrics.

It builds review-level prediction records containing:

- actual label
- predicted label
- positive probability
- confidence
- prediction type

It then analyzes:

- false positives
- false negatives
- high-confidence mistakes
- low-confidence / near-boundary cases
- errors by review length
- global model weights
- per-review feature contributions

## 10.1 Error counts

Recorded baseline errors:

```text
False positives: 1,802
False negatives: 1,761
```

## 10.2 Global sentiment features

Because the model is linear, each TF-IDF feature has a directly inspectable learned weight.

Examples of strongly positive features include:

```text
great
excellent
wonderful
perfect
best
amazing
favorite
loved
love
enjoyed
fantastic
brilliant
```

Examples of strongly negative features include:

```text
worst
bad
awful
waste
worse
poor
boring
terrible
nothing
horrible
no
poorly
stupid
dull
crap
```

## 10.3 Per-review explanation

For a linear TF-IDF classifier, the contribution of a feature can be written as:

```text
feature contribution = TF-IDF value × learned model weight
```

The notebook uses this to identify which words contribute most strongly to an individual prediction.

This is one of the main advantages of the traditional model: its decisions are much easier to inspect than those of the Transformer.

---

# 11. Controlled n-gram experiment

The traditional model is then improved through a controlled experiment.

The question:

> Does adding short phrase information improve sentiment classification?

Two alternatives are compared:

```text
Unigram:          ngram_range=(1, 1)
Unigram + Bigram: ngram_range=(1, 2)
```

All other important settings remain fixed.

Shared vectorizer settings include:

```text
lowercase=True
min_df=2
max_df=0.95
max_features=20,000
sublinear_tf=True
dtype=float32
```

Training settings remain:

```text
batch size=128
epochs=8
learning rate=1e-3
weight decay=1e-4
```

Randomness is reset before each experiment.

## 11.1 Experiment results

| Experiment | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 |
|---|---:|---:|---:|---:|
| Unigram | 0.866425 | 0.855033 | 0.883889 | 0.869222 |
| **Unigram + Bigram** | **0.880347** | **0.868012** | **0.898353** | **0.882922** |

Difference:

```text
Δ Validation F1       = +0.0137
Δ Validation Accuracy = +0.0139
```

Relative validation F1 improvement reported by the notebook:

```text
1.58%
```

The unigram+bigram configuration wins and becomes the selected traditional model.

## 11.2 Representation cost

Unigram training matrix:

```text
NNZ:     2,580,136
Density: 0.6508%
Vectorization time: ~3.28 s
```

Unigram+bigram training matrix:

```text
NNZ:     4,128,409
Density: 1.0413%
Vectorization time: ~11.74 s
```

Training time remained similar in the recorded run, while feature extraction became more expensive.

---

# 12. Best traditional model

Selected configuration:

```text
TF-IDF
ngram_range=(1, 2)
max_features=20,000
+
PyTorch Logistic Regression
```

Saved validation information:

```text
Best epoch:         8
Best validation F1: 0.882922
```

## 12.1 Final test result

The selected model is loaded from saved artifacts and evaluated without additional fitting.

Test matrix:

```text
Shape: (25,000, 20,000)
NNZ:   5,084,941
```

Final metrics:

| Metric | Value |
|---|---:|
| Accuracy | **0.87624** |
| Precision | 0.868805 |
| Recall | **0.88632** |
| F1 | **0.877475** |
| Macro F1 | 0.8762 |

This becomes the strongest final model in the recorded project results.

---

# 13. Transformer approach

The Transformer branch uses:

```text
distilbert-base-uncased
```

with a binary sequence-classification head.

The project explicitly works through the mechanics of Transformer input rather than treating the model as a black box.

Topics covered include:

- subword tokenization
- token IDs
- special tokens
- padding
- dynamic padding
- attention masks
- truncation
- PyTorch tensors
- pretrained weights
- transfer learning
- classification logits
- softmax probabilities
- fine-tuning

---

# 14. Transformer input representation

A review is processed approximately as:

```text
raw review
   ↓
tokenizer
   ↓
subword tokens
   ↓
input_ids
attention_mask
   ↓
DistilBERT
```

For a batch, the main tensors have shapes similar to:

```text
input_ids:      [batch_size, sequence_length]
attention_mask: [batch_size, sequence_length]
labels:         [batch_size]
```

Dynamic padding is used so examples in the same batch are padded to a common length, while truncation enforces the maximum input length.

---

# 15. DistilBERT fine-tuning configuration

Final training configuration used in the notebook:

```text
Model:                 distilbert-base-uncased
Number of labels:      2
Maximum length:        128
Train batch size:      16
Validation batch size: 32
Optimizer:             AdamW
Learning rate:         2e-5
Gradient clipping:     max norm 1.0
Epochs:                2
```

The saved checkpoint is selected by validation F1.

## 15.1 Training history

Epoch 1:

```text
Train Loss:          0.3666
Validation Loss:     0.3117
Validation Accuracy: 0.8684
Validation F1:       0.8710
Epoch Time:          120.2 s
```

Epoch 2:

```text
Train Loss:          0.2417
Validation Loss:     0.3063
Validation Accuracy: 0.8745
Validation F1:       0.8737
Epoch Time:          121.0 s
```

Best checkpoint:

```text
Best epoch:         2
Best validation F1: 0.8737
Total training time: 241.7 s
```

The recorded run used CUDA.

---

# 16. DistilBERT final test result

The best saved model and tokenizer are reloaded before test evaluation.

Final metrics:

| Metric | Value |
|---|---:|
| Test loss | 0.311351 |
| Accuracy | 0.873560 |
| Precision | **0.880842** |
| Recall | 0.864000 |
| F1 | 0.872340 |
| Macro F1 | 0.873548 |

Recorded test inference time:

```text
39.13 seconds for 25,000 reviews
```

As with all timing results, this is environment-dependent and should not be compared across machines without a controlled benchmark.

---

# 17. Final model comparison

Final comparison recorded in `07_model_comparison.ipynb`:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **TF-IDF + Logistic Regression** | **0.87624** | 0.868805 | **0.88632** | **0.877475** |
| DistilBERT | 0.87356 | **0.880842** | 0.86400 | 0.872340 |

Winner by test F1:

```text
TF-IDF + Logistic Regression
```

Difference reported in the comparison notebook:

```text
DistilBERT Δ F1       = -0.0051
DistilBERT Δ Accuracy = -0.0027
```

The traditional model wins F1, accuracy, and recall in this recorded setup, while DistilBERT achieves higher positive-class precision.

---

# 18. Interpreting the result

The result does **not** prove that TF-IDF is generally better than Transformers.

It only shows that, under the exact configuration tested here, the optimized traditional model obtained a slightly higher test F1 than the DistilBERT baseline.

Several project-specific factors are plausible contributors.

## 18.1 Strong lexical sentiment signals

Movie reviews contain highly informative words and short phrases.

Examples:

```text
great
excellent
terrible
waste
not good
very bad
```

TF-IDF with bigrams can represent many of these efficiently.

## 18.2 Truncation

The Transformer baseline uses:

```text
max_length = 128
```

The EDA shows:

```text
median review length = 174 words
95th percentile      = 595 words
99th percentile      ≈ 909 words
```

Tokens are not identical to words, but the EDA still makes it clear that many reviews are substantially longer than the Transformer input limit used here.

Therefore, truncation is a plausible limitation.

This project does not experimentally isolate truncation as the cause of the result.

## 18.3 Minimal Transformer tuning

The DistilBERT branch intentionally stays close to a baseline:

```text
2 epochs
learning rate 2e-5
max length 128
one model family
```

There is no extensive search over:

- context length
- learning rate
- scheduler
- warmup
- number of epochs
- layer freezing
- gradient accumulation
- model architecture

Therefore, the Transformer result should be interpreted as a useful baseline rather than a fully optimized upper bound.

## 18.4 Complexity vs engineering value

The traditional model is:

- fast to train
- cheap to run
- sparse
- highly interpretable
- simple to deploy

DistilBERT provides much stronger contextual modeling, but has higher training and inference cost.

The practical lesson is:

> Select a model using the full engineering objective, not model sophistication alone.

---

# 19. Model trade-offs

| Property | TF-IDF + Logistic Regression | DistilBERT |
|---|---|---|
| Representation | Sparse statistical features | Contextual token representations |
| Word order/context | Limited; improved by n-grams | Strong |
| Training cost | Low | High |
| Inference cost | Low | Higher |
| Interpretability | High | Lower |
| Deployment complexity | Low | Higher |
| Final test F1 | **0.8775** | 0.8723 |

---

# 20. Inference pipeline

The project includes a final inference notebook using the saved DistilBERT artifacts.

Single-text pipeline:

```text
new review
   ↓
saved tokenizer
   ↓
input_ids + attention_mask
   ↓
saved DistilBERT
   ↓
2 logits
   ↓
softmax
   ↓
negative / positive probabilities
   ↓
predicted label
```

The notebook also supports batch inference.

## 20.1 Recorded positive example

Input:

```text
This movie was absolutely fantastic. I loved every minute of it.
```

Recorded output:

```text
Prediction: pos
Negative probability: 0.0171
Positive probability: 0.9829
```

## 20.2 Recorded negative example

The notebook also evaluates:

```text
The movie was boring, predictable, and painfully slow. I hated it.
```

Recorded probabilities:

```text
Negative: 0.991905
Positive: 0.008095
```

## 20.3 Context-sensitive examples

The inference notebook includes examples such as:

```text
I thought this movie would be terrible, but it was actually surprisingly good.
```

Recorded prediction:

```text
pos
P(pos) ≈ 0.9608
```

and:

```text
The movie started really well, but the ending was so bad that I cannot recommend it.
```

Recorded prediction:

```text
neg
P(neg) ≈ 0.9913
```

These examples are demonstrations, not a separate benchmark.

---

# 21. Repository structure

A clean repository layout for this project is:

```text
.
├── README.md
│
├── docs/
│   └── PROJECT_DETAILS.md
│
├── data/
│   ├── huggingface_cache/
│   └── processed/
│       └── imdb_clean_splits/
│
├── models/
│   ├── best_traditional/
│   │   ├── tfidf_logistic_regression.pt
│   │   └── tfidf_vectorizer.joblib
│   │
│   └── distilbert_imdb/
│       ├── config files
│       ├── model weights
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
│   ├── data_cleaning/
│   ├── baseline/
│   ├── best_traditional/
│   ├── distilbert/
│   ├── tfidf_experiments/
│   └── final_comparison/
│
├── scripts/
│   └── prepare_clean_data.py
│
└── src/
    └── predict_tfidf.py
```

If your local notebook filenames still contain temporary suffixes such as `(1)` or `(2)`, rename them to the clean repository names shown above before committing.

---

# 22. Notebook guide

## `01_eda.ipynb`

Purpose:

- load the cleaned dataset
- verify split sizes
- inspect label distribution
- check missing/empty text
- check duplicates and overlap
- analyze review lengths
- quantify HTML content
- save EDA summary information

Important outputs:

```text
Train:      19,823
Validation: 4,956
Test:       25,000
HTML reviews: 58.77%
Median words: 174
```

---

## `02_tfidf_pytorch_baseline.ipynb`

Purpose:

- mild text cleanup
- majority baseline
- TF-IDF unigram representation
- sparse mini-batch handling
- PyTorch Logistic Regression
- validation metrics
- best-checkpoint selection
- artifact saving

Important result:

```text
Best validation F1: 0.8687
```

---

## `03_tfidf_test_evaluation.ipynb`

Purpose:

- load saved baseline artifacts
- transform the official test set
- batched inference
- final baseline metrics
- classification report
- confusion matrix
- error analysis
- feature interpretation
- per-review contribution analysis
- save detailed reports

Important result:

```text
Test F1: 0.8577
```

---

## `04_tfidf_experiments.ipynb`

Purpose:

- controlled unigram vs unigram+bigram comparison
- keep training settings fixed
- reset randomness
- compare validation metrics
- inspect sparsity and feature composition
- select the best traditional configuration

Important result:

```text
Unigram F1:          0.8692
Unigram+Bigram F1:   0.8829
Δ F1:               +0.0137
```

---

## `05_best_traditional_test.ipynb`

Purpose:

- load selected unigram+bigram model
- verify vectorizer/model compatibility
- evaluate once without retraining
- produce final traditional metrics

Important result:

```text
Test F1: 0.8775
```

---

## `06_distilbert_baseline.ipynb`

Purpose:

- explain Transformer tokenization
- inspect tokens and IDs
- demonstrate special tokens
- demonstrate padding and attention masks
- demonstrate truncation
- build Transformer Dataset/DataLoader
- inspect one real batch
- demonstrate one training step
- fine-tune DistilBERT
- select best validation checkpoint
- perform final Transformer test evaluation

Important results:

```text
Best validation F1: 0.8737
Test F1:            0.8723
```

---

## `07_model_comparison.ipynb`

Purpose:

- load final metrics
- compare traditional and Transformer models
- calculate F1 and accuracy deltas
- summarize engineering trade-offs

Important result:

```text
Best model: TF-IDF + Logistic Regression
Best test F1: 0.8775
```

---

## `08_inference_demo.ipynb`

Purpose:

- load the saved DistilBERT model and tokenizer
- predict sentiment for new text
- return probabilities
- batch multiple reviews
- derive a simple confidence score
- run sanity checks

---

# 23. Generated reports

The notebooks save analysis outputs instead of keeping all important results only in memory.

Depending on the notebook, generated reports include files such as:

```text
test_metrics.csv
classification_report.csv
confusion_matrix.csv
test_predictions.csv
false_positives.csv
false_negatives.csv
error_by_review_length.csv
feature_weights.csv
top_sentiment_features.csv
experiment_comparison.csv
training_histories.csv
model_metrics_comparison.csv
model_tradeoffs.csv
```

The exact filenames vary slightly between notebooks.

---

# 24. Installation

The notebooks use the following main Python libraries:

```text
torch
transformers
datasets
scikit-learn
pandas
numpy
scipy
matplotlib
joblib
jupyter
```

A simple environment can be created with:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy scipy matplotlib joblib jupyter
```

Start Jupyter:

```bash
jupyter lab
```

No exact dependency versions are documented in the provided project notebooks, so this documentation intentionally does not claim a fully pinned environment.

For stronger reproducibility, export a tested environment into `requirements.txt` or a lock file before publishing the repository.

---

# 25. Reproducing the project

Recommended order:

```text
1. Prepare / clean the dataset
2. Run 01_eda.ipynb
3. Run 02_tfidf_pytorch_baseline.ipynb
4. Run 03_tfidf_test_evaluation.ipynb
5. Run 04_tfidf_experiments.ipynb
6. Run 05_best_traditional_test.ipynb
7. Run 06_distilbert_baseline.ipynb
8. Run 07_model_comparison.ipynb
9. Run 08_inference_demo.ipynb
```

For a stricter untouched-test protocol in a future rerun, move exploratory error analysis off the official test split and onto validation data before model selection.

---

# 26. Model artifacts

## Traditional model

The selected traditional pipeline requires two artifacts:

```text
TF-IDF vectorizer
+
PyTorch model checkpoint
```

This is necessary because the PyTorch model only understands a fixed ordered feature vector. The fitted vectorizer stores the vocabulary, feature ordering, and IDF values that define those 20,000 input columns.

At inference time:

```text
raw review
→ same mild cleanup
→ saved vectorizer.transform(...)
→ model input vector
→ saved PyTorch classifier
```

The vectorizer must never be fitted again on validation, test, or new inference text.

## DistilBERT

The Transformer artifact directory contains both the fine-tuned model and tokenizer files.

At inference time:

```text
raw review
→ saved tokenizer
→ saved model
```

No fitting occurs.

---

# 27. Limitations

## 27.1 Official test-set exposure during development

As described earlier, the baseline test set is analyzed before the later bigram experiment.

This weakens the interpretation of the test set as a perfectly untouched final holdout.

## 27.2 Transformer truncation

The DistilBERT run uses:

```text
max_length=128
```

while many IMDb reviews are substantially longer.

Long-review information may therefore be discarded.

## 27.3 Limited Transformer tuning

Only one Transformer model and a small set of fine-tuning choices are explored.

The project does not establish the best achievable DistilBERT performance.

## 27.4 Binary labels only

The task contains only:

```text
negative
positive
```

Mixed, neutral, or ambiguous reviews are still forced into one of these classes.

## 27.5 Confidence is not calibration

The inference demo defines a simple confidence value as the maximum predicted class probability.

This should not automatically be interpreted as a perfectly calibrated probability of correctness.

## 27.6 Timing results are environment-dependent

Recorded vectorization, training, throughput, and inference times depend on the specific CPU/GPU environment used when the notebooks were executed.

They are useful for the project record but not universal benchmarks.

---

# 28. Future work

High-value next experiments include:

### Better Transformer treatment of long reviews

- increase maximum sequence length
- compare 128 vs 256 vs 512 tokens
- use chunking for long reviews
- aggregate chunk predictions

### Better fine-tuning

- learning-rate scheduler
- warmup
- 3+ epochs with validation monitoring
- gradient accumulation
- mixed-precision training
- early stopping

### Stronger evaluation protocol

- keep the official test split completely locked
- perform all error analysis on validation
- repeat runs with multiple random seeds
- report mean and standard deviation

### Transformer error analysis

- inspect high-confidence Transformer mistakes
- compare disagreement cases between TF-IDF and DistilBERT
- analyze performance by review length
- analyze negation and contrastive phrases

### Probability quality

- reliability diagrams
- expected calibration error
- temperature scaling

### Deployment

- expose inference through a REST API
- benchmark CPU latency and memory
- quantize the Transformer
- compare deployment cost of traditional and Transformer models
- build a small interactive UI

---

# 29. Key lessons

This project demonstrates several important machine-learning principles.

## Start with a strong baseline

A simple model can be competitive when the representation matches the task well.

## Keep sparse data sparse

The TF-IDF matrix is extremely sparse. Converting the entire matrix to dense form would be unnecessary and memory-inefficient.

## Validation should drive model selection

The bigram configuration was selected because it improved validation F1 while the remaining training settings were held constant.

## Interpretability can guide experiments

Linear feature weights and review-level contributions make it possible to understand what the traditional model has learned.

## Pretraining changes the optimization problem

DistilBERT does not learn language from IMDb from scratch. Fine-tuning adjusts pretrained language representations for sentiment classification.

## Context comes at a cost

Transformers model context much more naturally than TF-IDF, but require substantially more compute and introduce sequence-length constraints.

## More complex is not always better

Under the exact configuration tested here:

```text
TF-IDF + Logistic Regression F1 = 0.8775
DistilBERT F1                  = 0.8723
```

The simpler system wins the recorded comparison.

That result is not a universal statement about model families. It is evidence that the correct engineering decision depends on the dataset, evaluation protocol, resource constraints, and the amount of tuning performed.

---

# 30. Final summary

The completed project contains two end-to-end NLP pipelines.

### Traditional

```text
IMDb review
→ cleanup
→ TF-IDF unigram+bigram
→ sparse mini-batches
→ PyTorch Logistic Regression
→ probability
→ sentiment
```

### Transformer

```text
IMDb review
→ tokenizer
→ input IDs + attention mask
→ fine-tuned DistilBERT
→ logits
→ softmax
→ sentiment
```

Final recorded result:

```text
Best model:
TF-IDF + Logistic Regression

Accuracy: 0.87624
F1:       0.877475
```

The value of the project is not only the final score. It shows the full path from dataset hygiene and classical NLP to modern Transformer fine-tuning, while keeping the comparison measurable, inspectable, and engineering-focused.
