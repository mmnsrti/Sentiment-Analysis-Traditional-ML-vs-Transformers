from pathlib import Path

from datasets import DatasetDict, load_dataset


CACHE_DIR = Path("data/huggingface_cache")
OUTPUT_DIR = Path("data/processed/imdb_splits")
RANDOM_SEED = 42


def main() -> None:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("Loading IMDb dataset...")

    dataset = load_dataset(
        "stanfordnlp/imdb",
        cache_dir=str(CACHE_DIR),
    )

    print("\nOriginal dataset:")
    print(dataset)

    train_validation = dataset["train"].train_test_split(
        test_size=0.20,
        seed=RANDOM_SEED,
        stratify_by_column="label",
    )

    final_dataset = DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": dataset["test"],
        }
    )

    print("\nFinal dataset splits:")
    print(final_dataset)

    print("\nSplit sizes:")
    print(f"Train: {len(final_dataset['train']):,}")
    print(f"Validation: {len(final_dataset['validation']):,}")
    print(f"Test: {len(final_dataset['test']):,}")

    print("\nLabel names:")
    label_names = final_dataset["train"].features["label"].names

    for label_id, label_name in enumerate(label_names):
        print(f"{label_id} = {label_name}")

    print("\nExample:")
    example = final_dataset["train"][0]

    print(f"Label ID: {example['label']}")
    print(f"Label: {label_names[example['label']]}")
    print(f"Text:\n{example['text'][:500]}")

    final_dataset.save_to_disk(str(OUTPUT_DIR))

    print(f"\nDataset saved successfully in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()