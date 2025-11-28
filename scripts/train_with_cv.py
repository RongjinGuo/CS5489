"""
Training script with proper 3-fold CV on train set, then full training
Follows the rubric requirement: CV on train set for hyperparameter tuning
"""
import argparse
import yaml
import torch
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import BPETokenizer, WordTokenizer, MTDataset, collate_fn
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.gru_seq2seq import GRUSeq2Seq
from src.models.transformer import TransformerModel
from src.trainer import Trainer
from torch.utils.data import DataLoader, Subset
from scripts.train import build_tokenizers, build_model


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cv_on_train(train_dataset, config, model_type, device, src_tokenizer, tgt_tokenizer, n_folds=3):
    """Run 3-fold CV on training set to select best hyperparameters"""
    print(f"\n{'='*60}")
    print(f"Step 1: {n_folds}-Fold Cross-Validation on Training Set")
    print(f"{'='*60}\n")
    
    # Hyperparameter grid
    hyperparams_list = [
        {'training.learning_rate': 0.0001, 'model.rnn.hidden_size': 256, 'model.rnn.dropout': 0.1},
        {'training.learning_rate': 0.0001, 'model.rnn.hidden_size': 512, 'model.rnn.dropout': 0.1},
        {'training.learning_rate': 0.0005, 'model.rnn.hidden_size': 256, 'model.rnn.dropout': 0.2},
        {'training.learning_rate': 0.0005, 'model.rnn.hidden_size': 512, 'model.rnn.dropout': 0.2},
        {'training.learning_rate': 0.001, 'model.rnn.hidden_size': 256, 'model.rnn.dropout': 0.3},
        {'training.learning_rate': 0.001, 'model.rnn.hidden_size': 512, 'model.rnn.dropout': 0.3},
    ]
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = np.arange(len(train_dataset))
    
    all_results = []
    
    # Get vocab sizes once (tokenizers are already built)
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    for hp_idx, hyperparams in enumerate(hyperparams_list):
        print(f"\nHyperparameter set {hp_idx + 1}/{len(hyperparams_list)}: {hyperparams}")
        fold_losses = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            print(f"  Fold {fold_idx + 1}/{n_folds}...")
            
            # Create fold datasets
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(train_dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'],
                                    shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'],
                                  shuffle=False, collate_fn=collate_fn)
            
            # Update config with hyperparameters (deep copy to avoid modifying original)
            import copy
            fold_config = copy.deepcopy(config)
            for key, value in hyperparams.items():
                keys = key.split('.')
                d = fold_config
                for k in keys[:-1]:
                    d = d[k]
                d[keys[-1]] = value
            
            # Build model (vocab sizes are fixed, only hyperparams change)
            model = build_model(model_type, src_vocab_size, tgt_vocab_size, fold_config)
            
            # Train for fewer epochs during CV
            original_epochs = fold_config['training']['num_epochs']
            fold_config['training']['num_epochs'] = config.get('cross_validation', {}).get('cv_epochs', 5)
            fold_config['model_name'] = f"{model_type}_cv_fold{fold_idx+1}_hp{hp_idx+1}"
            
            trainer = Trainer(model, train_loader, val_loader, fold_config, device)
            trainer.train()
            
            fold_losses.append(trainer.best_val_loss)
            print(f"    Val Loss: {trainer.best_val_loss:.4f}")
            
            fold_config['training']['num_epochs'] = original_epochs
        
        mean_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        all_results.append({
            'hyperparams': hyperparams,
            'mean_val_loss': mean_loss,
            'std_val_loss': std_loss
        })
        print(f"  Mean Val Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    
    # Select best hyperparameters
    best_result = min(all_results, key=lambda x: x['mean_val_loss'])
    print(f"\n{'='*60}")
    print("Best Hyperparameters from CV:")
    print(f"{'='*60}")
    print(f"{best_result['hyperparams']}")
    print(f"Mean Val Loss: {best_result['mean_val_loss']:.4f} ± {best_result['std_val_loss']:.4f}")
    
    return best_result['hyperparams']


def main():
    parser = argparse.ArgumentParser(description="Train MT model with CV")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "transformer"],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/iwslt17",
                       help="Path to data directory")
    parser.add_argument("--skip_cv", action="store_true",
                       help="Skip CV and use default hyperparameters")
    parser.add_argument("--n_folds", type=int, default=3,
                       help="Number of folds for CV")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['model_name'] = args.model
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build tokenizers
    src_tokenizer, tgt_tokenizer = build_tokenizers(config, args.data_dir)
    
    # Build training dataset (for CV)
    data_dir = Path(args.data_dir)
    train_dataset = MTDataset(
        data_dir / "train" / "train.en",
        data_dir / "train" / "train.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    # Step 1: Cross-validation on train set
    if not args.skip_cv:
        best_hyperparams = run_cv_on_train(train_dataset, config, args.model, device, 
                                          src_tokenizer, tgt_tokenizer, args.n_folds)
        
        # Update config with best hyperparameters
        for key, value in best_hyperparams.items():
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
    else:
        print("Skipping CV, using default hyperparameters")
    
    # Step 2: Train on full training set with best hyperparameters
    print(f"\n{'='*60}")
    print("Step 2: Training on Full Training Set")
    print(f"{'='*60}\n")
    
    # Full training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Validation set (separate from train)
    val_dataset = MTDataset(
        data_dir / "valid" / "valid.en",
        data_dir / "valid" / "valid.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Build model with best hyperparameters
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    model = build_model(args.model, src_vocab_size, tgt_vocab_size, config)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nBest model saved to: {trainer.save_dir}/")
    print(f"Training history: {trainer.save_dir}/history.json")


if __name__ == "__main__":
    main()

