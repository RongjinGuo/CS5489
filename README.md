# Machine Translation Project: English → Chinese

This project implements multiple neural machine translation models for English to Chinese translation, including LSTM, GRU, and Transformer architectures.

## Project Structure

```
MT/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── download_data.py
│   └── prepare_iwslt17.py
├── src/
│   ├── __init__.py
│   ├── data_exploration.py      # Data analysis and visualization
│   ├── preprocessing.py          # Tokenization and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_seq2seq.py      # LSTM Encoder-Decoder
│   │   ├── gru_seq2seq.py       # GRU Encoder-Decoder
│   │   └── transformer.py       # Transformer model
│   ├── trainer.py                # Training loop
│   ├── evaluator.py              # BLEU evaluation
│   └── visualization.py          # TSNE and clustering
├── scripts/
│   ├── train.py                  # Main training script
│   └── evaluate.py               # Evaluation script
├── notebooks/
│   └── data_exploration.ipynb    # Jupyter notebook for exploration
└── report/
    └── report_template.md        # Report template
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
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt
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

- Word-level tokenization
- BPE (Byte Pair Encoding) subword tokenization
- Hyperparameter tuning
- TSNE visualization of embeddings
- BLEU score evaluation

