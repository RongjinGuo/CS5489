"""
Transformer model for Machine Translation
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for machine translation"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt_input, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: [batch_size, src_len]
            tgt_input: [batch_size, tgt_len]
            src_key_padding_mask: [batch_size, src_len] (True for padding)
            tgt_key_padding_mask: [batch_size, tgt_len] (True for padding)
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size]
        """
        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_len, d_model]
        tgt_emb = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)  # [batch_size, tgt_len, d_model]
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        
        # Positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        # Create tgt_mask for causal masking
        tgt_len = tgt_emb.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt_emb.device)
        
        # Encode
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Decode
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Transpose back and project
        output = output.transpose(0, 1)  # [batch_size, tgt_len, d_model]
        logits = self.output_proj(output)  # [batch_size, tgt_len, tgt_vocab_size]
        
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src, src_key_padding_mask=None):
        """Encode source sentences"""
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb.transpose(0, 1)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

