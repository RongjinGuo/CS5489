"""
Evaluation module for MT models
"""
import torch
from sacrebleu import BLEU
from tqdm import tqdm
import numpy as np
from src.beam_search import BeamSearchDecoder


class Evaluator:
    """Evaluator for machine translation models"""
    
    def __init__(self, model, tokenizer, device, max_length=128, beam_size=5, use_beam_search=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.beam_size = beam_size
        self.use_beam_search = use_beam_search
        if use_beam_search:
            self.beam_decoder = BeamSearchDecoder(model, beam_size=beam_size, max_length=max_length)
    
    def translate(self, src_sentence, src_tokenizer, tgt_tokenizer):
        """Translate a single sentence"""
        self.model.eval()
        
        # Encode source
        src_ids = src_tokenizer.encode(src_sentence, add_bos=False, add_eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(self.device)
        
        # Use beam search if enabled
        if self.use_beam_search:
            with torch.no_grad():
                decoded_ids, _ = self.beam_decoder.decode(
                    src_tensor, src_lengths, src_tokenizer, tgt_tokenizer, self.device
                )
            return decoded_ids
        
        # Get encoder output
        if 'Transformer' in self.model.__class__.__name__:
            src_mask = (src_tensor == 0)
            encoder_output = self.model.encode(src_tensor, src_key_padding_mask=src_mask)
            # For transformer, we need to decode step by step
            return self._greedy_decode_transformer(src_tensor, encoder_output, tgt_tokenizer, src_mask)
        else:
            encoder_output, encoder_hidden = self.model.encode(src_tensor, src_lengths)
            return self._greedy_decode_rnn(encoder_output, encoder_hidden, src_lengths, tgt_tokenizer)
    
    def _greedy_decode_rnn(self, encoder_output, encoder_hidden, src_lengths, tgt_tokenizer):
        """Greedy decode for RNN models"""
        batch_size = 1
        tgt_vocab_size = self.model.decoder.output_proj.out_features
        
        # Start with BOS token (BPE uses id 2 for BOS)
        bos_id = 2 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<bos>', 2) if hasattr(tgt_tokenizer, 'special_tokens') else 2)
        tgt_input = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)
        decoded = []
        
        eos_id = 3 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<eos>', 3) if hasattr(tgt_tokenizer, 'special_tokens') else 3)
        
        for _ in range(self.max_length):
            if 'LSTM' in self.model.__class__.__name__:
                logits = self.model.decoder(tgt_input, encoder_output, encoder_hidden, src_lengths)
            else:  # GRU
                logits = self.model.decoder(tgt_input, encoder_output, encoder_hidden, src_lengths)
            
            # Get next token
            next_token = logits[:, -1, :].argmax(dim=-1)
            decoded.append(next_token.item())
            
            # Check for EOS
            if next_token.item() == eos_id:
                break
            
            # Update input
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
        
        return decoded
    
    def _greedy_decode_transformer(self, src, memory, tgt_tokenizer, src_mask):
        """Greedy decode for Transformer"""
        batch_size = 1
        bos_id = 2 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<bos>', 2) if hasattr(tgt_tokenizer, 'special_tokens') else 2)
        tgt_input = torch.tensor([[bos_id]], dtype=torch.long).to(self.device)
        decoded = []
        
        eos_id = 3 if hasattr(tgt_tokenizer, 'sp') else (tgt_tokenizer.special_tokens.get('<eos>', 3) if hasattr(tgt_tokenizer, 'special_tokens') else 3)
        
        for _ in range(self.max_length):
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            tgt_padding_mask = (tgt_input == 0)
            
            logits = self.model(
                src,
                tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            next_token = logits[:, -1, :].argmax(dim=-1)
            decoded.append(next_token.item())
            
            if next_token.item() == eos_id:
                break
            
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
        
        return decoded
    
    def evaluate_bleu(self, test_loader, src_tokenizer, tgt_tokenizer):
        """Evaluate BLEU score on test set"""
        self.model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                src_texts = batch['src_text']
                tgt_texts = batch['tgt_text']
                
                for src_text, tgt_text in zip(src_texts, tgt_texts):
                    # Translate
                    decoded_ids = self.translate(src_text, src_tokenizer, tgt_tokenizer)
                    
                    # Decode
                    # Both WordTokenizer and BPETokenizer have decode method
                    pred_text = tgt_tokenizer.decode(decoded_ids)
                    
                    predictions.append(pred_text)
                    references.append([tgt_text])  # BLEU expects list of references
        
        # Calculate BLEU
        bleu = BLEU()
        score = bleu.corpus_score(predictions, references)
        
        return score.score, predictions, references
    
    def show_examples(self, test_loader, src_tokenizer, tgt_tokenizer, n_examples=5):
        """Show translation examples"""
        self.model.eval()
        
        examples = []
        count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                src_texts = batch['src_text']
                tgt_texts = batch['tgt_text']
                
                for src_text, tgt_text in zip(src_texts, tgt_texts):
                    if count >= n_examples:
                        break
                    
                    decoded_ids = self.translate(src_text, src_tokenizer, tgt_tokenizer)
                    pred_text = tgt_tokenizer.decode(decoded_ids)
                    
                    examples.append({
                        'source': src_text,
                        'target': tgt_text,
                        'prediction': pred_text
                    })
                    count += 1
                
                if count >= n_examples:
                    break
        
        return examples

