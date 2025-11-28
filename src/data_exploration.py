import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

import matplotlib

# Use modern matplotlib style (seaborn-v0_8 is deprecated)
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback to default style if seaborn-v0_8 is not available
    plt.style.use('default')
sns.set_palette("husl")

# ====== 这里新增：全局中文字体设置 ======
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',   # 如果你装了 fonts-noto-cjk
    'WenQuanYi Zen Hei',  # 如果你装了文泉驿
    'SimHei'              # 万一系统有黑体
]
matplotlib.rcParams['axes.unicode_minus'] = False

# Use modern matplotlib style (seaborn-v0_8 is deprecated)
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    # Fallback to default style if seaborn-v0_8 is not available
    plt.style.use('default')
sns.set_palette("husl")

def load_data(data_dir, split="train"):
    """Load English and Chinese sentences"""
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    with open(split_dir / f"{split}.en", "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]
    
    with open(split_dir / f"{split}.zh", "r", encoding="utf-8") as f:
        zh_lines = [line.strip() for line in f]
    
    return en_lines, zh_lines

def analyze_length_distribution(en_lines, zh_lines, save_path="figures/length_distribution.png"):
    """Analyze and plot sentence length distribution"""
    en_lengths = [len(line.split()) for line in en_lines]
    zh_lengths = [len(line) for line in zh_lines]  # Chinese: character count
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # English word length
    axes[0].hist(en_lengths, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of Words', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('English Sentence Length Distribution', fontsize=14)
    axes[0].axvline(np.mean(en_lengths), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(en_lengths):.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Chinese character length
    axes[1].hist(zh_lengths, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Number of Characters', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Chinese Sentence Length Distribution', fontsize=14)
    axes[1].axvline(np.mean(zh_lengths), color='r', linestyle='--',
                    label=f'Mean: {np.mean(zh_lengths):.1f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved length distribution to {save_path}")
    
    return {
        'en_mean': np.mean(en_lengths),
        'en_std': np.std(en_lengths),
        'en_max': np.max(en_lengths),
        'zh_mean': np.mean(zh_lengths),
        'zh_std': np.std(zh_lengths),
        'zh_max': np.max(zh_lengths)
    }

def analyze_word_frequency(en_lines, zh_lines, top_k=30, save_path="figures/word_frequency.png"):
    """Analyze and plot word frequency"""
    # English word frequency
    en_words = []
    for line in en_lines:
        en_words.extend(line.lower().split())
    
    en_word_freq = Counter(en_words)
    en_top_words = en_word_freq.most_common(top_k)
    
    # Chinese character frequency
    zh_chars = []
    for line in zh_lines:
        zh_chars.extend(list(line))
    
    zh_char_freq = Counter(zh_chars)
    zh_top_chars = zh_char_freq.most_common(top_k)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # English
    words, freqs = zip(*en_top_words)
    axes[0].barh(range(len(words)), freqs, alpha=0.7)
    axes[0].set_yticks(range(len(words)))
    axes[0].set_yticklabels(words)
    axes[0].set_xlabel('Frequency', fontsize=12)
    axes[0].set_title(f'Top {top_k} English Words', fontsize=14)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Chinese
    chars, freqs = zip(*zh_top_chars)
    axes[1].barh(range(len(chars)), freqs, alpha=0.7, color='orange')
    axes[1].set_yticks(range(len(chars)))
    axes[1].set_yticklabels(chars, fontsize=10)
    axes[1].set_xlabel('Frequency', fontsize=12)
    axes[1].set_title(f'Top {top_k} Chinese Characters', fontsize=14)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved word frequency to {save_path}")
    
    return en_word_freq, zh_char_freq

def show_alignment_examples(en_lines, zh_lines, n_examples=5, save_path="figures/alignment_examples.txt"):
    """Show sentence alignment examples"""
    examples = []
    for i in range(min(n_examples, len(en_lines))):
        examples.append(f"Example {i+1}:")
        examples.append(f"English: {en_lines[i]}")
        examples.append(f"Chinese: {zh_lines[i]}")
        examples.append("")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(examples))
    
    print(f"Saved alignment examples to {save_path}")
    print("\n".join(examples[:20]))  # Print first few

def dataset_statistics(data_dir):
    """Print dataset statistics"""
    splits = ["train", "valid", "test"]
    stats = {}
    
    for split in splits:
        try:
            en_lines, zh_lines = load_data(data_dir, split)
            stats[split] = {
                'size': len(en_lines),
                'en_avg_length': np.mean([len(line.split()) for line in en_lines]),
                'zh_avg_length': np.mean([len(line) for line in zh_lines])
            }
        except FileNotFoundError:
            print(f"Warning: {split} set not found")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    for split, stat in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Size: {stat['size']:,} pairs")
        print(f"  Avg English length: {stat['en_avg_length']:.1f} words")
        print(f"  Avg Chinese length: {stat['zh_avg_length']:.1f} characters")
    
    return stats

def main():
    """Run all data exploration"""
    data_dir = "data/iwslt17"
    
    print("Loading data...")
    en_lines, zh_lines = load_data(data_dir, "train")
    
    print(f"Loaded {len(en_lines)} sentence pairs")
    
    # Statistics
    stats = dataset_statistics(data_dir)
    
    # Length distribution
    print("\nAnalyzing length distribution...")
    length_stats = analyze_length_distribution(en_lines, zh_lines)
    
    # Word frequency
    print("\nAnalyzing word frequency...")
    en_freq, zh_freq = analyze_word_frequency(en_lines, zh_lines)
    
    # Alignment examples
    print("\nShowing alignment examples...")
    show_alignment_examples(en_lines, zh_lines)
    
    print("\nData exploration completed!")

if __name__ == "__main__":
    main()

