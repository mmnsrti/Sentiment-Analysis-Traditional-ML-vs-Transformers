from pathlib import Path

from datasets import load_dataset


def main() -> None:
    data_dir = Path("data")
    cache_dir = data_dir / "huggingface_cache"

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "stanfordnlp/imdb",
        cache_dir=str(cache_dir),
    )
    dataset["train"].to_csv(
        data_dir / "imdb_train.csv",
        index=False,
    )

    dataset["test"].to_csv(
        data_dir / "imdb_test.csv",
        index=False,
    )
    print(dataset)

    print("\nFirst training example:")
    print("Label:", dataset["train"][0]["label"])
    print("Text:", dataset["train"][0]["text"][:500])

    print("\nNumber of samples:")
    print("Train:", len(dataset["train"]))
    print("Test:", len(dataset["test"]))


if __name__ == "__main__":
    main()