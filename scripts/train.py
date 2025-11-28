"""
Main training script
"""
import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import BPETokenizer, MTDataset, collate_fn
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.gru_seq2seq import GRUSeq2Seq
from src.models.transformer import TransformerModel
from src.trainer import Trainer
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_tokenizers(config, data_dir):
    """Build source and target tokenizers"""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    
    # Load training sentences
    with open(train_dir / "train.en", "r", encoding="utf-8") as f:
        en_sentences = [line.strip() for line in f]
    
    with open(train_dir / "train.zh", "r", encoding="utf-8") as f:
        zh_sentences = [line.strip() for line in f]
    
    tokenization = config['preprocessing']['tokenization']
    vocab_size = config['preprocessing']['vocab_size']
    
    if tokenization == "bpe":
        # Train BPE models
        print("Training BPE tokenizers...")
        src_tokenizer = BPETokenizer()
        src_tokenizer.train(en_sentences, vocab_size=vocab_size, 
                           model_prefix=f"{config['preprocessing']['bpe_model_prefix']}_en")
        
        tgt_tokenizer = BPETokenizer()
        tgt_tokenizer.train(zh_sentences, vocab_size=vocab_size,
                           model_prefix=f"{config['preprocessing']['bpe_model_prefix']}_zh")
        
        print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
        print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    elif tokenization == "word":
        # Word-level tokenization
        print("Building word-level tokenizers...")
        from src.preprocessing import WordTokenizer
        
        src_tokenizer = WordTokenizer(vocab_size=vocab_size, min_freq=2)
        src_tokenizer.build_vocab(en_sentences)
        
        tgt_tokenizer = WordTokenizer(vocab_size=vocab_size, min_freq=2)
        tgt_tokenizer.build_vocab(zh_sentences)
        
        print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
        print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")
    else:
        raise ValueError(f"Unknown tokenization: {tokenization}. Use 'word' or 'bpe'")
    
    return src_tokenizer, tgt_tokenizer


def build_model(model_type, src_vocab_size, tgt_vocab_size, config):
    """Build model based on type"""
    if model_type == "lstm":
        model = LSTMSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=256,
            hidden_dim=config['model']['rnn']['hidden_size'],
            num_layers=config['model']['rnn']['num_layers'],
            dropout=config['model']['rnn']['dropout'],
            bidirectional=config['model']['rnn']['bidirectional']
        )
    elif model_type == "gru":
        model = GRUSeq2Seq(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=256,
            hidden_dim=config['model']['rnn']['hidden_size'],
            num_layers=config['model']['rnn']['num_layers'],
            dropout=config['model']['rnn']['dropout'],
            bidirectional=config['model']['rnn']['bidirectional']
        )
    elif model_type == "transformer":
        model = TransformerModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['model']['transformer']['d_model'],
            nhead=config['model']['transformer']['nhead'],
            num_encoder_layers=config['model']['transformer']['num_encoder_layers'],
            num_decoder_layers=config['model']['transformer']['num_decoder_layers'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MT model")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "transformer"],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/iwslt17",
                       help="Path to data directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build tokenizers
    src_tokenizer, tgt_tokenizer = build_tokenizers(config, args.data_dir)
    
    # Build datasets
    data_dir = Path(args.data_dir)
    train_dataset = MTDataset(
        data_dir / "train" / "train.en",
        data_dir / "train" / "train.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    val_dataset = MTDataset(
        data_dir / "valid" / "valid.en",
        data_dir / "valid" / "valid.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Build model
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    model = build_model(args.model, src_vocab_size, tgt_vocab_size, config)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Add model name to config for saving
    config['model_name'] = args.model
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Train
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()

