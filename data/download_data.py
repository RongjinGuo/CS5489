"""
Process manually downloaded en-zh dataset
Supports multiple formats: TSV, CSV, separate files, etc.
"""
import zipfile
import os
from pathlib import Path
import pandas as pd
import csv
import shutil


def process_en_zh_dataset():
    """Process manually downloaded en-zh.zip file"""
    data_dir = Path("data/iwslt17")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = Path("data/en-zh.zip")
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found!")
        print("Please make sure en-zh.zip is in the data/ directory")
        return
    
    print(f"Processing {zip_path}...")
    
    # Extract zip file to a temporary directory
    extract_dir = Path("data/en-zh_extracted")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(exist_ok=True)
    
    print("Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✓ Extraction completed")
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return
    
    # Find all files
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if not file.startswith('.'):
                all_files.append(Path(root) / file)
    
    print(f"Found {len(all_files)} files")
    
    # Try different processing methods
    success = False
    
    # Method 1: Look for TSV/CSV files with parallel data
    tsv_csv_files = [f for f in all_files if f.suffix.lower() in ['.tsv', '.csv', '.txt']]
    if tsv_csv_files:
        print(f"\nTrying to process as TSV/CSV files...")
        success = process_as_tsv_csv(tsv_csv_files, data_dir)
        if success:
            cleanup(extract_dir)
            return
    
    # Method 2: Look for separate en/zh files
    en_files = [f for f in all_files if any(x in f.name.lower() for x in ['en', 'english', 'src'])]
    zh_files = [f for f in all_files if any(x in f.name.lower() for x in ['zh', 'chinese', 'cn', 'tgt', 'target'])]
    
    if en_files and zh_files:
        print(f"\nFound separate English and Chinese files")
        print(f"  English: {[f.name for f in en_files[:3]]}")
        print(f"  Chinese: {[f.name for f in zh_files[:3]]}")
        success = process_separate_files(en_files, zh_files, data_dir)
        if success:
            cleanup(extract_dir)
            return
    
    # Method 3: If only 2 files, treat as parallel
    if len(tsv_csv_files) == 2:
        print(f"\nFound 2 files, treating as parallel text...")
        success = process_two_files(tsv_csv_files, data_dir)
        if success:
            cleanup(extract_dir)
            return
    
    # Method 4: Look for train/valid/test structure
    train_files = [f for f in all_files if 'train' in f.name.lower()]
    valid_files = [f for f in all_files if any(x in f.name.lower() for x in ['valid', 'dev', 'val'])]
    test_files = [f for f in all_files if 'test' in f.name.lower()]
    
    if train_files:
        print(f"\nFound train/valid/test structure")
        success = process_pre_split_files(train_files, valid_files, test_files, data_dir)
        if success:
            cleanup(extract_dir)
            return
    
    # Method 5: Try to read any text file as TSV (tab-separated)
    print(f"\nTrying to read files as tab-separated...")
    for f in tsv_csv_files[:3]:
        success = try_read_as_tsv(f, data_dir)
        if success:
            cleanup(extract_dir)
            return
    
    print("\n" + "="*60)
    print("Could not automatically process the dataset.")
    print("="*60)
    print(f"\nFiles found in zip:")
    for f in all_files[:20]:
        print(f"  {f.relative_to(extract_dir)}")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")
    print("\nPlease check the file structure and update the script if needed.")
    print(f"Extracted files are in: {extract_dir}")
    cleanup(extract_dir)


def process_as_tsv_csv(files, data_dir):
    """Process TSV/CSV files"""
    all_pairs = []
    
    for f in files:
        print(f"  Reading {f.name}...")
        try:
            # Try TSV first (tab-separated)
            try:
                df = pd.read_csv(f, sep='\t', header=None, encoding='utf-8', 
                               on_bad_lines='skip', quoting=csv.QUOTE_NONE, 
                               dtype=str, na_filter=False)
            except:
                # Try CSV (comma-separated)
                df = pd.read_csv(f, sep=',', header=None, encoding='utf-8',
                               on_bad_lines='skip', quoting=csv.QUOTE_NONE,
                               dtype=str, na_filter=False)
            
            # Check if it has at least 2 columns
            if df.shape[1] < 2:
                print(f"    Skipping {f.name} (only {df.shape[1]} columns)")
                continue
            
            # Read pairs
            for idx, row in df.iterrows():
                if len(row) >= 2:
                    en = str(row.iloc[0]).strip()
                    zh = str(row.iloc[1]).strip()
                    if en and zh and len(en) > 0 and len(zh) > 0:
                        all_pairs.append((en, zh))
            
            print(f"    Found {len(all_pairs)} pairs so far")
            
        except Exception as e:
            print(f"    Error reading {f.name}: {e}")
            continue
    
    if not all_pairs:
        return False
    
    print(f"\n✓ Total pairs found: {len(all_pairs):,}")
    split_and_save(all_pairs, data_dir)
    return True


def process_separate_files(en_files, zh_files, data_dir):
    """Process separate English and Chinese files"""
    # Read English files
    all_en = []
    for f in sorted(en_files):
        print(f"  Reading {f.name}...")
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                lines = [line.strip() for line in file if line.strip()]
                all_en.extend(lines)
                print(f"    {len(lines)} lines")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Read Chinese files
    all_zh = []
    for f in sorted(zh_files):
        print(f"  Reading {f.name}...")
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                lines = [line.strip() for line in file if line.strip()]
                all_zh.extend(lines)
                print(f"    {len(lines)} lines")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Match lengths
    min_len = min(len(all_en), len(all_zh))
    if min_len == 0:
        print("Error: No valid sentences found")
        return False
    
    print(f"\n✓ Matched {min_len:,} sentence pairs")
    pairs = list(zip(all_en[:min_len], all_zh[:min_len]))
    split_and_save(pairs, data_dir)
    return True


def process_two_files(files, data_dir):
    """Process two files as parallel text"""
    if len(files) != 2:
        return False
    
    print(f"  Reading {files[0].name} and {files[1].name}...")
    
    try:
        with open(files[0], 'r', encoding='utf-8', errors='ignore') as f1:
            lines1 = [line.strip() for line in f1 if line.strip()]
        
        with open(files[1], 'r', encoding='utf-8', errors='ignore') as f2:
            lines2 = [line.strip() for line in f2 if line.strip()]
        
        min_len = min(len(lines1), len(lines2))
        if min_len == 0:
            return False
        
        print(f"✓ Found {min_len:,} parallel lines")
        pairs = list(zip(lines1[:min_len], lines2[:min_len]))
        split_and_save(pairs, data_dir)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def process_pre_split_files(train_files, valid_files, test_files, data_dir):
    """Process files that are already split"""
    # This is a simplified version - may need adjustment based on actual structure
    print("Note: Pre-split files detected, but automatic pairing may be needed")
    return False  # For now, let other methods handle it


def try_read_as_tsv(file_path, data_dir):
    """Try to read a single file as TSV"""
    print(f"  Trying {file_path.name}...")
    try:
        # Try tab-separated
        df = pd.read_csv(file_path, sep='\t', header=None, nrows=100, 
                        encoding='utf-8', on_bad_lines='skip', dtype=str)
        if df.shape[1] >= 2:
            # Read full file
            df = pd.read_csv(file_path, sep='\t', header=None,
                           encoding='utf-8', on_bad_lines='skip', dtype=str)
            pairs = []
            for _, row in df.iterrows():
                if len(row) >= 2:
                    en = str(row.iloc[0]).strip()
                    zh = str(row.iloc[1]).strip()
                    if en and zh:
                        pairs.append((en, zh))
            
            if pairs:
                print(f"✓ Found {len(pairs):,} pairs")
                split_and_save(pairs, data_dir)
                return True
    except:
        pass
    return False


def split_and_save(pairs, data_dir):
    """Split pairs into train/valid/test and save"""
    # Shuffle for random split
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # Split: 80% train, 10% valid, 10% test
    n_total = len(pairs)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]
    
    # Save train
    train_dir = data_dir / "train"
    train_dir.mkdir(exist_ok=True)
    with open(train_dir / "train.en", "w", encoding="utf-8") as f:
        f.write("\n".join([p[0] for p in train_pairs]) + "\n")
    with open(train_dir / "train.zh", "w", encoding="utf-8") as f:
        f.write("\n".join([p[1] for p in train_pairs]) + "\n")
    print(f"✓ Saved {len(train_pairs):,} training pairs")
    
    # Save valid
    val_dir = data_dir / "valid"
    val_dir.mkdir(exist_ok=True)
    with open(val_dir / "valid.en", "w", encoding="utf-8") as f:
        f.write("\n".join([p[0] for p in val_pairs]) + "\n")
    with open(val_dir / "valid.zh", "w", encoding="utf-8") as f:
        f.write("\n".join([p[1] for p in val_pairs]) + "\n")
    print(f"✓ Saved {len(val_pairs):,} validation pairs")
    
    # Save test
    test_dir = data_dir / "test"
    test_dir.mkdir(exist_ok=True)
    with open(test_dir / "test.en", "w", encoding="utf-8") as f:
        f.write("\n".join([p[0] for p in test_pairs]) + "\n")
    with open(test_dir / "test.zh", "w", encoding="utf-8") as f:
        f.write("\n".join([p[1] for p in test_pairs]) + "\n")
    print(f"✓ Saved {len(test_pairs):,} test pairs")
    
    print("\n" + "="*60)
    print("Dataset processing completed successfully!")
    print("="*60)
    print(f"Data saved to: {data_dir}")
    print(f"Total pairs: {n_total:,}")
    print(f"  Train: {len(train_pairs):,} ({len(train_pairs)/n_total*100:.1f}%)")
    print(f"  Valid: {len(val_pairs):,} ({len(val_pairs)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_pairs):,} ({len(test_pairs)/n_total*100:.1f}%)")


def cleanup(extract_dir):
    """Clean up extracted files"""
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            print(f"\nCleaned up temporary files")
    except:
        pass


if __name__ == "__main__":
    process_en_zh_dataset()
