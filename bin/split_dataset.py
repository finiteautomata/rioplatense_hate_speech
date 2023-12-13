import fire
from datasets import load_dataset


def split_dataset(dataset_name, num_splits=10, output_dir="data", random_state=42):
    dataset = load_dataset(dataset_name, split="test")
    df = dataset.to_pandas()

    # Shuffle dataset

    df = df.sample(frac=1, random_state=random_state)

    del df["body"]

    split_size = len(df) // num_splits

    for i in range(num_splits):
        split = df.iloc[i * split_size : (i + 1) * split_size]
        # Zero fill split number
        path = f"{output_dir}/test_{str(i+1).zfill(2)}.csv"
        split.to_csv(path, index=False)

        print(f"Saved split {i} to {path}")


if __name__ == "__main__":
    fire.Fire(split_dataset)
