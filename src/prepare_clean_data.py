from __future__ import annotations

from html import unescape
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Value,
    load_dataset,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_cache"

# Save under a new name so we do not confuse it with the old leaky split.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "imdb_clean_splits"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "data_cleaning"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20


def normalize_for_deduplication(text: str) -> str:
    """
    Create a normalized version used ONLY for duplicate detection.

    Important:
    We preserve the original review text for model training.
    This normalized text is only a comparison key.
    """

    # Convert unusual Unicode variants into a consistent representation.
    text = unicodedata.normalize("NFKC", str(text))

    # Convert things such as &amp; into their normal characters.
    text = unescape(text)

    # Replace IMDb line-break HTML with a space.
    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove any remaining HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert repeated spaces/newlines/tabs into one space.
    text = re.sub(r"\s+", " ", text)

    # casefold is similar to lowercase but stronger for text comparison.
    return text.strip().casefold()


def create_huggingface_dataset(
    dataframe: pd.DataFrame,
    features: Features,
) -> Dataset:
    """
    Convert a pandas DataFrame back into a Hugging Face Dataset.
    """

    clean_dataframe = (
        dataframe[["text", "label"]]
        .reset_index(drop=True)
        .copy()
    )

    return Dataset.from_pandas(
        clean_dataframe,
        features=features,
        preserve_index=False,
    )


def calculate_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> int:
    left_keys = set(left["_dedup_key"])
    right_keys = set(right["_dedup_key"])

    return len(left_keys.intersection(right_keys))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("Loading the original IMDb dataset...")

    raw_dataset = load_dataset(
        "stanfordnlp/imdb",
        cache_dir=str(CACHE_DIR),
    )

    original_train = (
        raw_dataset["train"]
        .to_pandas()[["text", "label"]]
        .copy()
    )

    official_test = (
        raw_dataset["test"]
        .to_pandas()[["text", "label"]]
        .copy()
    )

    print(f"Original train rows: {len(original_train):,}")
    print(f"Official test rows: {len(official_test):,}")

    # ---------------------------------------------------------
    # 1. Create duplicate-detection keys
    # ---------------------------------------------------------

    original_train["_dedup_key"] = original_train["text"].map(
        normalize_for_deduplication
    )

    official_test["_dedup_key"] = official_test["text"].map(
        normalize_for_deduplication
    )

    # ---------------------------------------------------------
    # 2. Detect duplicates with conflicting labels
    # ---------------------------------------------------------
    # Example of a conflicting duplicate:
    # Exactly the same review is labeled Positive once
    # and Negative somewhere else.

    labels_per_text = original_train.groupby(
        "_dedup_key"
    )["label"].nunique()

    conflicting_keys = set(
        labels_per_text[
            labels_per_text > 1
        ].index
    )

    conflicting_rows = original_train[
        original_train["_dedup_key"].isin(conflicting_keys)
    ].copy()

    print(
        "Conflicting duplicate groups:",
        len(conflicting_keys),
    )

    print(
        "Rows belonging to conflicting groups:",
        len(conflicting_rows),
    )

    if not conflicting_rows.empty:
        conflicting_rows[
            ["text", "label"]
        ].to_csv(
            REPORTS_DIR / "conflicting_label_examples.csv",
            index=False,
        )

    # Drop all versions of conflicting examples.
    # We cannot confidently decide which label is correct.
    train_without_conflicts = original_train[
        ~original_train["_dedup_key"].isin(conflicting_keys)
    ].copy()

    # ---------------------------------------------------------
    # 3. Remove internal duplicates
    # ---------------------------------------------------------

    duplicate_mask = train_without_conflicts.duplicated(
        subset=["_dedup_key"],
        keep=False,
    )

    duplicate_examples = train_without_conflicts[
        duplicate_mask
    ].copy()

    if not duplicate_examples.empty:
        duplicate_examples[
            ["text", "label"]
        ].to_csv(
            REPORTS_DIR / "duplicate_examples.csv",
            index=False,
        )

    rows_before_internal_dedup = len(train_without_conflicts)

    unique_train_pool = train_without_conflicts.drop_duplicates(
        subset=["_dedup_key"],
        keep="first",
    ).copy()

    internal_duplicates_removed = (
        rows_before_internal_dedup
        - len(unique_train_pool)
    )

    print(
        "Internal duplicate rows removed:",
        internal_duplicates_removed,
    )

    # ---------------------------------------------------------
    # 4. Decontaminate training data against official test data
    # ---------------------------------------------------------
    # We keep the official test set unchanged.
    # If a training review is also present in test,
    # remove its training copy.

    test_keys = set(official_test["_dedup_key"])

    train_test_overlap_mask = unique_train_pool[
        "_dedup_key"
    ].isin(test_keys)

    train_test_overlap_examples = unique_train_pool[
        train_test_overlap_mask
    ].copy()

    if not train_test_overlap_examples.empty:
        train_test_overlap_examples[
            ["text", "label"]
        ].to_csv(
            REPORTS_DIR / "train_test_overlap_examples.csv",
            index=False,
        )

    train_test_rows_removed = int(
        train_test_overlap_mask.sum()
    )

    clean_train_pool = unique_train_pool[
        ~train_test_overlap_mask
    ].copy()

    print(
        "Training rows removed because they also appear in test:",
        train_test_rows_removed,
    )

    # ---------------------------------------------------------
    # 5. Create a new stratified train/validation split
    # ---------------------------------------------------------

    train_df, validation_df = train_test_split(
        clean_train_pool,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=clean_train_pool["label"],
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    official_test = official_test.reset_index(drop=True)

    # sklearn's stratify argument preserves approximately
    # the same class ratio in both splits.
    # random_state makes the split reproducible.

    # ---------------------------------------------------------
    # 6. Final leakage checks
    # ---------------------------------------------------------

    train_validation_overlap = calculate_overlap(
        train_df,
        validation_df,
    )

    train_test_overlap = calculate_overlap(
        train_df,
        official_test,
    )

    validation_test_overlap = calculate_overlap(
        validation_df,
        official_test,
    )

    train_internal_duplicates = train_df.duplicated(
        subset=["_dedup_key"]
    ).sum()

    validation_internal_duplicates = validation_df.duplicated(
        subset=["_dedup_key"]
    ).sum()

    print("\nFinal checks")
    print("-------------------------------")
    print(
        "Train internal duplicates:",
        train_internal_duplicates,
    )
    print(
        "Validation internal duplicates:",
        validation_internal_duplicates,
    )
    print(
        "Train–Validation overlap:",
        train_validation_overlap,
    )
    print(
        "Train–Test overlap:",
        train_test_overlap,
    )
    print(
        "Validation–Test overlap:",
        validation_test_overlap,
    )

    # Stop immediately if leakage still exists.
    assert train_internal_duplicates == 0
    assert validation_internal_duplicates == 0
    assert train_validation_overlap == 0
    assert train_test_overlap == 0
    assert validation_test_overlap == 0

    # ---------------------------------------------------------
    # 7. Convert back into Hugging Face Dataset format
    # ---------------------------------------------------------

    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(
                names=["neg", "pos"]
            ),
        }
    )

    clean_dataset = DatasetDict(
        {
            "train": create_huggingface_dataset(
                train_df,
                features,
            ),
            "validation": create_huggingface_dataset(
                validation_df,
                features,
            ),
            "test": create_huggingface_dataset(
                official_test,
                features,
            ),
        }
    )

    # Safely replace only our newly generated output folder.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    clean_dataset.save_to_disk(str(OUTPUT_DIR))

    # ---------------------------------------------------------
    # 8. Save a cleaning report
    # ---------------------------------------------------------

    cleaning_report = pd.DataFrame(
        {
            "metric": [
                "original_train_rows",
                "conflicting_duplicate_groups",
                "conflicting_rows_removed",
                "internal_duplicate_rows_removed",
                "train_test_rows_removed",
                "final_train_rows",
                "final_validation_rows",
                "official_test_rows",
                "final_train_negative",
                "final_train_positive",
                "final_validation_negative",
                "final_validation_positive",
                "train_validation_overlap",
                "train_test_overlap",
                "validation_test_overlap",
            ],
            "value": [
                len(original_train),
                len(conflicting_keys),
                len(conflicting_rows),
                internal_duplicates_removed,
                train_test_rows_removed,
                len(train_df),
                len(validation_df),
                len(official_test),
                int((train_df["label"] == 0).sum()),
                int((train_df["label"] == 1).sum()),
                int((validation_df["label"] == 0).sum()),
                int((validation_df["label"] == 1).sum()),
                train_validation_overlap,
                train_test_overlap,
                validation_test_overlap,
            ],
        }
    )

    report_path = REPORTS_DIR / "cleaning_report.csv"

    cleaning_report.to_csv(
        report_path,
        index=False,
    )

    print("\nFinal dataset")
    print(clean_dataset)

    print("\nSaved cleaned dataset to:")
    print(OUTPUT_DIR)

    print("\nSaved cleaning report to:")
    print(report_path)


if __name__ == "__main__":
    main()