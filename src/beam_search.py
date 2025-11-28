"""
Beam search decoding for sequence generation
"""
import torch
import torch.nn.functional as F
import numpy as np


class BeamSearchDecoder:
    """Beam search decoder for sequence-to-sequence models"""
    
    def __init__(self, model, beam_size=5, max_length=128, length_penalty=0.6):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
    
    def decode(self, src, src_lengths, src_tokenizer, tgt_tokenizer, device):
        """
        Decode using beam search
        
        Args:
            src: source sequence tensor [1, src_len]
            src_lengths: source lengths [1]
            src_tokenizer: source tokenizer
            tgt_tokenizer: target tokenizer
            device: device
        
        Returns:
            best_sequence: list of token indices
            score: log probability score
        """
        self.model.eval()
        
        # Get encoder output
        if 'Transformer' in self.model.__class__.__name__:
            src_mask = (src == 0)
            encoder_output = self.model.encode(src, src_key_padding_mask=src_mask)
            return self._beam_search_transformer(src, encoder_output, src_mask, tgt_tokenizer, device)
        else:
            encoder_output, encoder_hidden = self.model.encode(src, src_lengths)
            return self._beam_search_rnn(encoder_output, encoder_hidden, src_lengths, tgt_tokenizer, device)
    
    def _beam_search_rnn(self, encoder_output, encoder_hidden, src_lengths, tgt_tokenizer, device):
        """Beam search for RNN models"""
        # Get special tokens
        bos_id = 2 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<bos>', 2) if hasattr(tgt_tokenizer, 'special_tokens') else 2)
        eos_id = 3 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<eos>', 3) if hasattr(tgt_tokenizer, 'special_tokens') else 3)
        
        # Initialize beam: (sequence, score, hidden_state)
        beam = [([bos_id], 0.0, encoder_hidden)]
        
        for step in range(self.max_length):
            candidates = []
            
            for seq, score, hidden in beam:
                # If sequence already ended, keep it
                if seq[-1] == eos_id:
                    candidates.append((seq, score, hidden))
                    continue
                
                # Get next token predictions
                tgt_input = torch.tensor([seq], dtype=torch.long).to(device)
                
                if 'LSTM' in self.model.__class__.__name__:
                    logits = self.model.decoder(tgt_input, encoder_output, hidden, src_lengths)
                else:  # GRU
                    logits = self.model.decoder(tgt_input, encoder_output, hidden, src_lengths)
                
                # Get top-k predictions
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                top_k_log_probs, top_k_indices = torch.topk(log_probs, self.beam_size, dim=-1)
                
                # Expand beam
                for i in range(self.beam_size):
                    token_id = top_k_indices[0, i].item()
                    token_log_prob = top_k_log_probs[0, i].item()
                    
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    
                    # Update hidden state (simplified - in practice need to track properly)
                    candidates.append((new_seq, new_score, hidden))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_size]
            
            # Check if all sequences ended
            if all(seq[-1] == eos_id for seq, _, _ in beam):
                break
        
        # Return best sequence
        best_seq, best_score, _ = max(beam, key=lambda x: x[1])
        
        # Apply length penalty
        length = len(best_seq)
        if length > 0:
            score = best_score / (length ** self.length_penalty)
        else:
            score = best_score
        
        return best_seq, score
    
    def _beam_search_transformer(self, src, memory, src_mask, tgt_tokenizer, device):
        """Beam search for Transformer models"""
        bos_id = 2 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<bos>', 2) if hasattr(tgt_tokenizer, 'special_tokens') else 2)
        eos_id = 3 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<eos>', 3) if hasattr(tgt_tokenizer, 'special_tokens') else 3)
        
        # Initialize beam
        beam = [([bos_id], 0.0)]
        
        for step in range(self.max_length):
            candidates = []
            
            for seq, score in beam:
                if seq[-1] == eos_id:
                    candidates.append((seq, score))
                    continue
                
                tgt_input = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = self.model.generate_square_subsequent_mask(len(seq)).to(device)
                tgt_padding_mask = (tgt_input == 0)
                
                logits = self.model(
                    src,
                    tgt_input,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                )
                
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                top_k_log_probs, top_k_indices = torch.topk(log_probs, self.beam_size, dim=-1)
                
                for i in range(self.beam_size):
                    token_id = top_k_indices[0, i].item()
                    token_log_prob = top_k_log_probs[0, i].item()
                    
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    candidates.append((new_seq, new_score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_size]
            
            if all(seq[-1] == eos_id for seq, _ in beam):
                break
        
        best_seq, best_score = max(beam, key=lambda x: x[1])
        length = len(best_seq)
        if length > 0:
            score = best_score / (length ** self.length_penalty)
        else:
            score = best_score
        
        return best_seq, score

