"""
Evaluation script
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
from src.evaluator import Evaluator
from src.visualization import plot_translation_examples
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_tokenizers(config, data_dir="data/iwslt17"):
    """Load trained tokenizers (supports both word and BPE)"""
    tokenization = config['preprocessing']['tokenization']
    
    if tokenization == "bpe":
        bpe_prefix = config['preprocessing']['bpe_model_prefix']
        
        src_tokenizer = BPETokenizer()
        src_tokenizer.load(f"{bpe_prefix}_en.model")
        
        tgt_tokenizer = BPETokenizer()
        tgt_tokenizer.load(f"{bpe_prefix}_zh.model")
    elif tokenization == "word":
        from src.preprocessing import WordTokenizer
        from pathlib import Path
        
        # For word tokenizers, we need to rebuild from training data
        # Load training sentences to rebuild vocab
        data_dir = Path(data_dir)
        train_dir = data_dir / "train"
        
        with open(train_dir / "train.en", "r", encoding="utf-8") as f:
            en_sentences = [line.strip() for line in f]
        
        with open(train_dir / "train.zh", "r", encoding="utf-8") as f:
            zh_sentences = [line.strip() for line in f]
        
        vocab_size = config['preprocessing']['vocab_size']
        
        src_tokenizer = WordTokenizer(vocab_size=vocab_size, min_freq=2)
        src_tokenizer.build_vocab(en_sentences)
        
        tgt_tokenizer = WordTokenizer(vocab_size=vocab_size, min_freq=2)
        tgt_tokenizer.build_vocab(zh_sentences)
    else:
        raise ValueError(f"Unknown tokenization: {tokenization}. Use 'word' or 'bpe'")
    
    return src_tokenizer, tgt_tokenizer


def load_model(model_type, checkpoint_path, config, device, data_dir="data/iwslt17"):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocab sizes from tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(config, data_dir)
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
    parser = argparse.ArgumentParser(description="Evaluate MT model")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "transformer"],
                       help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/iwslt17",
                       help="Path to data directory")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(config, args.data_dir)
    
    # Load model
    model = load_model(args.model, args.checkpoint, config, device, args.data_dir)
    print(f"Loaded {args.model} model from {args.checkpoint}")
    
    # Build test dataset
    data_dir = Path(args.data_dir)
    test_dataset = MTDataset(
        data_dir / args.split / f"{args.split}.en",
        data_dir / args.split / f"{args.split}.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Evaluator
    evaluator = Evaluator(
        model,
        tgt_tokenizer,
        device,
        max_length=config['evaluation']['max_length'],
        beam_size=config['evaluation']['beam_size'],
        use_beam_search=config['evaluation'].get('use_beam_search', False)
    )
    
    # Evaluate BLEU
    print("Evaluating BLEU score...")
    bleu_score, predictions, references = evaluator.evaluate_bleu(
        test_loader, src_tokenizer, tgt_tokenizer
    )
    
    print(f"\nBLEU Score: {bleu_score:.2f}")
    
    # Show examples
    print("\nGenerating translation examples...")
    examples = evaluator.show_examples(test_loader, src_tokenizer, tgt_tokenizer, n_examples=10)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save BLEU score
    with open(results_dir / f"{args.model}_bleu.json", "w") as f:
        json.dump({"bleu_score": bleu_score}, f, indent=2)
    
    # Save examples
    plot_translation_examples(examples, results_dir / f"{args.model}_examples.txt")
    
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()

