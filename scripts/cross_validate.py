"""
Cross-validation script for MT models
"""
import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import BPETokenizer, MTDataset
from src.cross_validation import CrossValidator, generate_hyperparameter_grid
from scripts.train import build_tokenizers, build_model


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Cross-validate MT model")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "transformer"],
                       help="Model type to cross-validate")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/iwslt17",
                       help="Path to data directory")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    parser.add_argument("--use_hyperparams", action="store_true",
                       help="Use hyperparameter search from config")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config['model_name'] = args.model
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build tokenizers
    src_tokenizer, tgt_tokenizer = build_tokenizers(config, args.data_dir)
    
    # Build training dataset (use train set for CV)
    data_dir = Path(args.data_dir)
    train_dataset = MTDataset(
        data_dir / "train" / "train.en",
        data_dir / "train" / "train.zh",
        src_tokenizer,
        tgt_tokenizer,
        max_length=config['data']['max_length']
    )
    
    print(f"Total training samples: {len(train_dataset)}")
    
    # Generate hyperparameter grid
    if args.use_hyperparams:
        hyperparams_list = generate_hyperparameter_grid(config)
        print(f"\nGenerated {len(hyperparams_list)} hyperparameter combinations")
    else:
        hyperparams_list = [{}]  # Use default config
        print("\nUsing default hyperparameters (no search)")
    
    # Create cross-validator
    config['_model_type'] = args.model
    config['data_dir'] = args.data_dir
    
    validator = CrossValidator(
        model_class=None,  # We'll build models inside CV
        train_dataset=train_dataset,
        config=config,
        device=device,
        n_folds=args.n_folds
    )
    
    # Run cross-validation
    results = validator.cross_validate(hyperparams_list)
    
    print("\n" + "="*60)
    print("Cross-validation completed!")
    print("="*60)
    print(f"\nBest hyperparameters: {validator.best_hyperparams}")
    print(f"\nResults saved to: checkpoints/cv_results/")


if __name__ == "__main__":
    main()

