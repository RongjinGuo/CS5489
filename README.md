# Machine Translation Project: English → Chinese

This project implements multiple neural machine translation models for English to Chinese translation, including LSTM, GRU, and Transformer architectures.

## Project Structure

```
MT/
├── README.md                      # This file (English)
├── QUICKSTART.md                  # Quick start guide (Chinese)
├── EXPERIMENT_GUIDE.md            # Complete experiment guide (Chinese)
├── requirements.txt               # Python dependencies
├── config.yaml                    # Configuration file
├── run_all.py                     # Complete pipeline script
├── data/
│   ├── download_data.py           # Download IWSLT17 dataset
│   ├── prepare_iwslt17.py         # Prepare and split dataset
│   └── iwslt17/                   # Processed dataset
├── src/
│   ├── __init__.py
│   ├── data_exploration.py         # Data analysis and visualization
│   ├── preprocessing.py            # Tokenization (Word/BPE)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_seq2seq.py        # LSTM Encoder-Decoder
│   │   ├── gru_seq2seq.py         # GRU Encoder-Decoder
│   │   └── transformer.py         # Transformer model
│   ├── trainer.py                 # Training loop
│   ├── evaluator.py               # BLEU evaluation
│   ├── beam_search.py             # Beam search decoding
│   ├── label_smoothing.py         # Label smoothing loss
│   ├── cross_validation.py        # Cross-validation utilities
│   └── visualization.py           # t-SNE and clustering
├── scripts/
│   ├── train.py                   # Main training script
│   ├── train_with_cv.py           # Training with cross-validation
│   ├── evaluate.py                # Evaluation script
│   ├── visualize.py              # Visualization script
│   └── summarize_results.py       # Results summarization
├── checkpoints/                   # Model checkpoints (generated)
├── results/                       # Evaluation results (generated)
├── figures/                      # Visualization figures (generated)
└── report/
    └── report_template.md         # Report template
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

1. **Download and prepare data:**
```bash
python data/download_data.py
python data/prepare_iwslt17.py
```

2. **Explore data:**
```bash
python src/data_exploration.py
```

3. **Train models:**
```bash
python scripts/train.py --model lstm --config config.yaml
python scripts/train.py --model gru --config config.yaml
python scripts/train.py --model transformer --config config.yaml
```

4. **Evaluate:**
```bash
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --config config.yaml
```

### Full Experimental Pipeline (Rubric-based)

For complete experimental setup following the rubric requirements, see **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)**.

Key features:
- ✅ 3-fold Cross-Validation on training set
- ✅ Word-level vs BPE tokenization comparison
- ✅ Beam Search decoding
- ✅ Label Smoothing
- ✅ t-SNE visualization and clustering
- ✅ Comprehensive results summarization

## Models

- **LSTM Seq2Seq**: Basic encoder-decoder with LSTM
- **GRU Seq2Seq**: Encoder-decoder with GRU
- **Transformer**: Attention-based transformer model

## Features

- **Tokenization**: Word-level and BPE (Byte Pair Encoding) subword tokenization
- **Models**: LSTM, GRU, and Transformer architectures with attention mechanisms
- **Training**: Cross-validation for hyperparameter tuning, early stopping, label smoothing
- **Decoding**: Greedy decoding and beam search
- **Evaluation**: BLEU score evaluation with translation examples
- **Visualization**: t-SNE dimensionality reduction, KMeans clustering, training curves

