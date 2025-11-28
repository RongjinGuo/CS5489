"""
Cross-validation module for MT models
"""
import torch
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import json
from tqdm import tqdm
from src.trainer import Trainer
from src.evaluator import Evaluator


class CrossValidator:
    """K-fold Cross-validation for machine translation models"""
    
    def __init__(self, model_class, train_dataset, config, device, n_folds=5):
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        self.n_folds = n_folds
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        self.fold_results = []
        self.best_hyperparams = None
        self._src_tokenizer = None
        self._tgt_tokenizer = None
    
    def cross_validate(self, hyperparams_list=None):
        """
        Perform K-fold cross-validation
        
        Args:
            hyperparams_list: List of hyperparameter dictionaries to try
                            If None, uses config defaults
        """
        if hyperparams_list is None:
            hyperparams_list = [{}]  # Use default config
        
        print(f"\n{'='*60}")
        print(f"Starting {self.n_folds}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        all_fold_results = []
        
        # Split data into folds
        indices = np.arange(len(self.train_dataset))
        fold_splits = list(self.kfold.split(indices))
        
        for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{self.n_folds}")
            print(f"{'='*60}")
            print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
            
            # Create fold-specific datasets
            train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.train_dataset, val_indices)
            
            # Create data loaders
            from src.preprocessing import collate_fn
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                collate_fn=collate_fn
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                collate_fn=collate_fn
            )
            
            fold_results = []
            
            # Try different hyperparameters
            for hp_idx, hyperparams in enumerate(hyperparams_list):
                print(f"\n  Hyperparameter set {hp_idx + 1}/{len(hyperparams_list)}")
                print(f"  {hyperparams}")
                
                # Update config with hyperparameters (deep copy to avoid modifying original)
                import copy
                fold_config = copy.deepcopy(self.config)
                for key, value in hyperparams.items():
                    # Handle nested keys like 'model.rnn.hidden_size'
                    keys = key.split('.')
                    d = fold_config
                    for k in keys[:-1]:
                        d = d[k]
                    d[keys[-1]] = value
                
                # Build model with current hyperparameters
                # Use existing tokenizers (already built)
                model_type = self.config.get('_model_type', self.config.get('model_name', 'lstm'))
                
                # Import here to avoid circular imports
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent))
                from scripts.train import build_model
                
                # Get vocab sizes from tokenizers (reuse from outer scope if available)
                # For CV, we'll build tokenizers once at the start
                if not hasattr(self, '_src_tokenizer'):
                    from scripts.train import build_tokenizers
                    data_dir = Path(self.config.get('data_dir', 'data/iwslt17'))
                    self._src_tokenizer, self._tgt_tokenizer = build_tokenizers(fold_config, str(data_dir))
                
                src_vocab_size = self._src_tokenizer.get_vocab_size()
                tgt_vocab_size = self._tgt_tokenizer.get_vocab_size()
                
                model = build_model(
                    model_type,
                    src_vocab_size,
                    tgt_vocab_size,
                    fold_config
                )
                
                # Train model
                fold_config['model_name'] = f"{self.config.get('model_name', 'model')}_fold{fold_idx+1}_hp{hp_idx+1}"
                trainer = Trainer(model, train_loader, val_loader, fold_config, self.device)
                
                # Train for fewer epochs during CV (to save time)
                original_epochs = fold_config['training']['num_epochs']
                cv_epochs = fold_config.get('cross_validation', {}).get('cv_epochs', 5)
                fold_config['training']['num_epochs'] = min(cv_epochs, original_epochs)
                
                trainer.train()
                
                # Get final validation loss
                final_val_loss = trainer.val_losses[-1] if trainer.val_losses else float('inf')
                
                result = {
                    'fold': fold_idx + 1,
                    'hyperparams': hyperparams,
                    'val_loss': final_val_loss,
                    'train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
                    'best_val_loss': trainer.best_val_loss
                }
                
                fold_results.append(result)
                all_fold_results.append(result)
                
                print(f"  Fold {fold_idx + 1} - Val Loss: {final_val_loss:.4f}")
                
                # Restore original epochs
                fold_config['training']['num_epochs'] = original_epochs
        
        # Aggregate results
        self.aggregate_results(all_fold_results, hyperparams_list)
        
        return self.fold_results
    
    def aggregate_results(self, all_results, hyperparams_list):
        """Aggregate results across folds"""
        print(f"\n{'='*60}")
        print("Cross-Validation Results Summary")
        print(f"{'='*60}\n")
        
        # Group by hyperparameters
        hp_results = {}
        for result in all_results:
            hp_key = str(result['hyperparams'])
            if hp_key not in hp_results:
                hp_results[hp_key] = []
            hp_results[hp_key].append(result['val_loss'])
        
        # Calculate mean and std for each hyperparameter set
        summary = []
        for hp_key, losses in hp_results.items():
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            summary.append({
                'hyperparams': eval(hp_key) if hp_key != '{}' else {},
                'mean_val_loss': mean_loss,
                'std_val_loss': std_loss,
                'n_folds': len(losses)
            })
        
        # Sort by mean validation loss
        summary.sort(key=lambda x: x['mean_val_loss'])
        
        print("Hyperparameter Performance (sorted by mean validation loss):")
        print("-" * 60)
        for i, result in enumerate(summary):
            print(f"{i+1}. {result['hyperparams']}")
            print(f"   Mean Val Loss: {result['mean_val_loss']:.4f} ± {result['std_val_loss']:.4f}")
            print()
        
        # Best hyperparameters
        if summary:
            self.best_hyperparams = summary[0]['hyperparams']
            print(f"Best Hyperparameters: {self.best_hyperparams}")
            print(f"Mean Validation Loss: {summary[0]['mean_val_loss']:.4f} ± {summary[0]['std_val_loss']:.4f}")
        
        self.fold_results = summary
        
        # Save results
        self.save_cv_results(summary)
    
    def save_cv_results(self, summary):
        """Save cross-validation results"""
        save_dir = Path(self.config['training']['save_dir']) / 'cv_results'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = save_dir / 'cv_summary.json'
        with open(results_file, 'w') as f:
            json.dump({
                'n_folds': self.n_folds,
                'best_hyperparams': self.best_hyperparams,
                'summary': summary
            }, f, indent=2)
        
        print(f"\nCross-validation results saved to {results_file}")


def generate_hyperparameter_grid(config):
    """Generate hyperparameter grid from config"""
    if not config['training']['hyperparameter_search']['enabled']:
        return [{}]  # Return default config
    
    search_space = config['training']['hyperparameter_search']['search_space']
    
    # Generate all combinations
    import itertools
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    combinations = list(itertools.product(*values))
    
    hyperparams_list = []
    for combo in combinations:
        hp = {}
        for key, value in zip(keys, combo):
            # Map to config path
            if key == 'learning_rate':
                hp['training.learning_rate'] = value
            elif key == 'dropout':
                hp['model.rnn.dropout'] = value
                hp['model.transformer.dropout'] = value
            elif key == 'hidden_size':
                hp['model.rnn.hidden_size'] = value
        hyperparams_list.append(hp)
    
    return hyperparams_list

