"""
Visualization script for embeddings and training curves
"""
import argparse
import yaml
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import BPETokenizer, MTDataset, collate_fn
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.gru_seq2seq import GRUSeq2Seq
from src.models.transformer import TransformerModel
from src.visualization import extract_embeddings, visualize_tsne, visualize_clusters, plot_training_curves
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_type, checkpoint_path, config, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocab sizes from tokenizers
    bpe_prefix = config['preprocessing']['bpe_model_prefix']
    src_tokenizer = BPETokenizer()
    src_tokenizer.load(f"{bpe_prefix}_en.model")
    tgt_tokenizer = BPETokenizer()
    tgt_tokenizer.load(f"{bpe_prefix}_zh.model")
    
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    # Build model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings and training curves")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "transformer"],
                       help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/iwslt17",
                       help="Path to data directory")
    parser.add_argument("--task", type=str, default="tsne", choices=["tsne", "cluster", "curves", "all"],
                       help="Visualization task")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, args.checkpoint, config, device)
    print(f"Loaded {args.model} model")
    
    # Load tokenizers and dataset
    bpe_prefix = config['preprocessing']['bpe_model_prefix']
    src_tokenizer = BPETokenizer()
    src_tokenizer.load(f"{bpe_prefix}_en.model")
    tgt_tokenizer = BPETokenizer()
    tgt_tokenizer.load(f"{bpe_prefix}_zh.model")
    
    data_dir = Path(args.data_dir)
    dataset = MTDataset(
        data_dir / "train" / "train.en",
        data_dir / "train" / "train.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Visualizations
    if args.task in ["tsne", "all"]:
        print("Extracting embeddings for t-SNE...")
        embeddings = extract_embeddings(
            model, data_loader, device,
            n_samples=config['visualization']['tsne']['n_samples']
        )
        
        visualize_tsne(
            embeddings,
            save_path=f"figures/{args.model}_tsne.png",
            perplexity=config['visualization']['tsne']['perplexity'],
            n_iter=config['visualization']['tsne']['n_iter']
        )
    
    if args.task in ["cluster", "all"]:
        print("Extracting embeddings for clustering...")
        embeddings = extract_embeddings(
            model, data_loader, device,
            n_samples=config['visualization']['tsne']['n_samples']
        )
        
        visualize_clusters(
            embeddings,
            n_clusters=5,
            save_path=f"figures/{args.model}_clusters.png"
        )
    
    if args.task in ["curves", "all"]:
        print("Plotting training curves...")
        checkpoint_dir = Path(args.checkpoint).parent
        history_path = checkpoint_dir / "history.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            plot_training_curves(
                history,
                save_path=f"figures/{args.model}_training_curves.png"
            )
        else:
            print(f"History file not found at {history_path}")
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()

