"""
Prepare IWSLT17 dataset: split train/valid/test if needed
"""
import os
from pathlib import Path
import random

def split_dataset(train_file_en, train_file_zh, output_dir, train_ratio=0.9, val_ratio=0.05):
    """Split dataset into train/valid/test"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    with open(train_file_en, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]
    
    with open(train_file_zh, "r", encoding="utf-8") as f:
        zh_lines = [line.strip() for line in f]
    
    assert len(en_lines) == len(zh_lines), "Mismatched number of lines"
    
    # Shuffle
    pairs = list(zip(en_lines, zh_lines))
    random.seed(42)
    random.shuffle(pairs)
    
    # Split
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]
    
    # Write splits
    splits = {
        "train": train_pairs,
        "valid": val_pairs,
        "test": test_pairs
    }
    
    for split_name, split_pairs in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        en_lines = [pair[0] + "\n" for pair in split_pairs]
        zh_lines = [pair[1] + "\n" for pair in split_pairs]
        
        with open(split_dir / f"{split_name}.en", "w", encoding="utf-8") as f:
            f.writelines(en_lines)
        
        with open(split_dir / f"{split_name}.zh", "w", encoding="utf-8") as f:
            f.writelines(zh_lines)
        
        print(f"{split_name}: {len(split_pairs)} pairs")
    
    print(f"Dataset split completed. Total: {n_total} pairs")

if __name__ == "__main__":
    data_dir = Path("data/iwslt17")
    train_dir = data_dir / "train"
    
    if (train_dir / "train.en").exists() and (train_dir / "train.zh").exists():
        # Check if already split
        if not (data_dir / "valid" / "valid.en").exists():
            print("Splitting dataset...")
            split_dataset(
                train_dir / "train.en",
                train_dir / "train.zh",
                data_dir
            )
        else:
            print("Dataset already split")
    else:
        print("Please run download_data.py first")

