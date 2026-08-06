from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

FILE_PATH = Path(__file__).resolve()

# This supports both of these locations:
#   project_root/predict_tfidf.py
#   project_root/src/predict_tfidf.py
PROJECT_ROOT = (
    FILE_PATH.parent.parent
    if FILE_PATH.parent.name == "src"
    else FILE_PATH.parent
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "tfidf_pytorch_logistic_regression.pt"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"


# ---------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------
# These rules intentionally match the training notebook.

HTML_BREAK_PATTERN = re.compile(
    r"<br\s*/?>",
    flags=re.IGNORECASE,
)

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text_for_model(text: str) -> str:
    """Apply the same mild cleanup used during model training."""

    text = unescape(str(text))
    text = HTML_BREAK_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


# ---------------------------------------------------------
# 3. Model architecture
# ---------------------------------------------------------
# The architecture must match the model used during training.

class LogisticRegressionModel(nn.Module):
    def __init__(self, number_of_features: int) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=number_of_features,
            out_features=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.linear(features)
        return logits.squeeze(dim=1)


# ---------------------------------------------------------
# 4. Load the saved vectorizer and model
# ---------------------------------------------------------

def load_artifacts(
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: torch.device | None = None,
) -> tuple[
    LogisticRegressionModel,
    TfidfVectorizer,
    torch.device,
]:
    """Load the fitted TF-IDF vectorizer and trained PyTorch model."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model_path = model_dir / MODEL_FILENAME
    vectorizer_path = model_dir / VECTORIZER_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}\n"
            "Run the baseline training notebook first."
        )

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer file was not found: {vectorizer_path}\n"
            "Run the baseline training notebook first."
        )

    vectorizer = joblib.load(vectorizer_path)

    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(
            "The loaded vectorizer is not a TfidfVectorizer."
        )

    checkpoint: dict[str, Any] = torch.load(
        model_path,
        map_location=device,
    )

    number_of_features = int(
        checkpoint["number_of_features"]
    )

    vectorizer_feature_count = len(
        vectorizer.get_feature_names_out()
    )

    if vectorizer_feature_count != number_of_features:
        raise ValueError(
            "The vectorizer and model are incompatible: "
            f"vectorizer has {vectorizer_feature_count:,} features, "
            f"but the model expects {number_of_features:,}."
        )

    model = LogisticRegressionModel(
        number_of_features=number_of_features
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, vectorizer, device


# ---------------------------------------------------------
# 5. Predict one review
# ---------------------------------------------------------

def predict_sentiment(
    text: str,
    model: LogisticRegressionModel,
    vectorizer: TfidfVectorizer,
    device: torch.device,
) -> dict[str, str | float]:
    """Predict the sentiment of one IMDb-style movie review."""

    cleaned_text = clean_text_for_model(text)

    if not cleaned_text:
        raise ValueError("The review is empty after cleaning.")

    # Use transform, not fit_transform.
    # We must preserve the vocabulary and IDF learned on training data.
    sparse_features = vectorizer.transform(
        [cleaned_text]
    )

    dense_features = sparse_features.toarray().astype(
        np.float32,
        copy=False,
    )

    feature_tensor = torch.from_numpy(
        dense_features
    ).to(device)

    with torch.no_grad():
        logit = model(feature_tensor)
        positive_probability = torch.sigmoid(
            logit
        ).item()

    negative_probability = 1.0 - positive_probability
    label_id = int(positive_probability >= 0.5)
    label = "pos" if label_id == 1 else "neg"

    return {
        "label": label,
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
    }


# ---------------------------------------------------------
# 6. Command-line interface
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict movie-review sentiment with the saved "
            "TF-IDF + PyTorch model."
        )
    )

    parser.add_argument(
        "review",
        nargs="?",
        help=(
            "Review text. If omitted, the program asks for it "
            "interactively."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing the saved .pt and .joblib files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    review = args.review

    if review is None:
        review = input("Enter a movie review: ").strip()

    model, vectorizer, device = load_artifacts(
        model_dir=args.model_dir
    )

    result = predict_sentiment(
        text=review,
        model=model,
        vectorizer=vectorizer,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Prediction: {result['label']}")
    print(
        "Positive probability: "
        f"{result['positive_probability']:.2%}"
    )
    print(
        "Negative probability: "
        f"{result['negative_probability']:.2%}"
    )


if __name__ == "__main__":
    main()