"""
Training module for MT models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json


class Trainer:
    """Trainer class for machine translation models"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
        
        # Loss function (ignore padding)
        use_label_smoothing = config['training'].get('label_smoothing', 0.0)
        if use_label_smoothing > 0:
            from src.label_smoothing import LabelSmoothingLoss
            # Get vocab size from model
            if hasattr(model, 'decoder'):
                if hasattr(model.decoder, 'output_proj'):
                    vocab_size = model.decoder.output_proj.out_features
                elif hasattr(model.decoder, 'out'):
                    vocab_size = model.decoder.out.out_features
                else:
                    vocab_size = 8000  # Default fallback
            elif hasattr(model, 'output_proj'):
                vocab_size = model.output_proj.out_features
            else:
                vocab_size = 8000  # Default fallback
            self.criterion = LabelSmoothingLoss(vocab_size, padding_idx=0, smoothing=use_label_smoothing)
            print(f"Using label smoothing with smoothing={use_label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        
        # Save directory - create subdirectory for this model if specified
        base_save_dir = Path(config['training']['save_dir'])
        model_name = config.get('model_name', 'model')
        self.save_dir = base_save_dir / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        for batch in pbar:
            # Move to device
            src = torch.tensor(batch['src'], dtype=torch.long).to(self.device)
            tgt_input = torch.tensor(batch['tgt_input'], dtype=torch.long).to(self.device)
            tgt_output = torch.tensor(batch['tgt_output'], dtype=torch.long).to(self.device)
            src_lengths = torch.tensor(batch['src_lengths'], dtype=torch.long).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, nn.Module) and hasattr(self.model, 'forward'):
                # Check if it's transformer
                if 'Transformer' in self.model.__class__.__name__:
                    # Create padding masks
                    src_mask = (src == 0)
                    tgt_mask = (tgt_input == 0)
                    logits = self.model(src, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
                else:
                    logits = self.model(src, tgt_input, src_lengths)
            else:
                logits = self.model(src, tgt_input, src_lengths)
            
            # Handle length mismatch: truncate tgt_output to match logits length
            # logits shape: [batch_size, tgt_len, vocab_size]
            # tgt_output shape: [batch_size, tgt_output_len]
            tgt_len = logits.size(1)
            tgt_output_len = tgt_output.size(1)
            
            if tgt_len != tgt_output_len:
                # Truncate or pad tgt_output to match logits
                if tgt_output_len > tgt_len:
                    tgt_output = tgt_output[:, :tgt_len]
                else:
                    # Pad tgt_output (shouldn't happen, but just in case)
                    padding = torch.zeros(tgt_output.size(0), tgt_len - tgt_output_len, 
                                        dtype=tgt_output.dtype, device=tgt_output.device)
                    tgt_output = torch.cat([tgt_output, padding], dim=1)
            
            # Reshape for loss
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = self.criterion(logits, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                src = torch.tensor(batch['src'], dtype=torch.long).to(self.device)
                tgt_input = torch.tensor(batch['tgt_input'], dtype=torch.long).to(self.device)
                tgt_output = torch.tensor(batch['tgt_output'], dtype=torch.long).to(self.device)
                src_lengths = torch.tensor(batch['src_lengths'], dtype=torch.long).to(self.device)
                
                if 'Transformer' in self.model.__class__.__name__:
                    src_mask = (src == 0)
                    tgt_mask = (tgt_input == 0)
                    logits = self.model(src, tgt_input, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
                else:
                    logits = self.model(src, tgt_input, src_lengths)
                
                # Handle length mismatch
                tgt_len = logits.size(1)
                tgt_output_len = tgt_output.size(1)
                if tgt_len != tgt_output_len:
                    if tgt_output_len > tgt_len:
                        tgt_output = tgt_output[:, :tgt_len]
                    else:
                        padding = torch.zeros(tgt_output.size(0), tgt_len - tgt_output_len, 
                                            dtype=tgt_output.dtype, device=tgt_output.device)
                        tgt_output = torch.cat([tgt_output, padding], dim=1)
                
                logits = logits.reshape(-1, logits.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = self.criterion(logits, tgt_output)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Full training loop"""
        patience = self.config['training'].get('early_stopping_patience', 5)
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint (with model name prefix)
            model_name = self.config.get('model_name', 'model')
            self.save_checkpoint(f'{model_name}_latest')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save training history
        self.save_history()
        self.save_training_log()
    
    def save_checkpoint(self, name='checkpoint'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'config': self.config
        }
        # If name already contains model name, use it as is; otherwise add model name prefix
        model_name = self.config.get('model_name', 'model')
        if model_name not in name:
            filename = f"{model_name}_{name}.pt"
        else:
            filename = f"{name}.pt"
        torch.save(checkpoint, self.save_dir / filename)
        if 'best' in name:
            print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f}) to {self.save_dir / filename}")
    
    def save_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'num_epochs': len(self.train_losses),
            'final_epoch': self.epoch
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to {self.save_dir / 'history.json'}")
    
    def save_training_log(self):
        """Save detailed training log to text file"""
        log_file = self.save_dir / 'training_log.txt'
        with open(log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Total Epochs: {len(self.train_losses)}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n\n")
            f.write("Epoch\tTrain Loss\tVal Loss\n")
            f.write("-" * 60 + "\n")
            for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
                f.write(f"{i+1}\t{train_loss:.4f}\t{val_loss:.4f}\n")
        print(f"Training log saved to {log_file}")

