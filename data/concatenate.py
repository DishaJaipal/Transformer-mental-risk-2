import pandas as pd
import os
from tqdm import tqdm


def prepare_data():
    """
    Concatenate train, dev, and test datasets
    """
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "datasets")
    os.makedirs(data_dir, exist_ok=True)

    # Load raw datasets
    train_df = pd.read_csv("train.csv")
    dev_df = pd.read_csv("dev.csv")
    test_df = pd.read_csv("test.csv")

    print(f"   Train: {len(train_df)} samples")
    print(f"   Dev:   {len(dev_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    print(f"   Total: {len(train_df) + len(dev_df) + len(test_df)} samples")

    # Concatenate all
    print("Concatenating datasets...")
    new_data = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    # Rename columns
    new_data.rename(
        columns={"PID": "pid", "Text_data": "text", "Label": "labels"}, inplace=True
    )

    # Drop null or empty text
    new_data = new_data.dropna(subset=["text"])
    new_data = new_data[new_data["text"].str.split().apply(len) >= 3]

    # Drop full-row duplicates
    new_data = new_data.drop_duplicates()

    # Save cleaned file
    output_path = os.path.join(data_dir, "combined_cleaned.csv")
    with tqdm(total=1, desc="Saving CSV") as pbar:
        new_data.to_csv(output_path, index=False)
        pbar.update(1)
    print(f"✓ Saved cleaned combined data to {output_path}")

    return new_data


if __name__ == "__main__":
    prepare_data()
