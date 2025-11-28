"""
Label smoothing loss for sequence-to-sequence models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size * seq_len, vocab_size]
            target: [batch_size * seq_len]
        Returns:
            loss: scalar
        """
        # Flatten
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for padding and true class
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        # Mask padding
        mask = (target != self.padding_idx)
        true_dist = true_dist * mask.unsqueeze(1).float()
        
        # KL divergence
        kl_div = F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='none')
        loss = kl_div.sum(dim=1) * mask.float()
        
        return loss.sum() / mask.sum().clamp(min=1)

