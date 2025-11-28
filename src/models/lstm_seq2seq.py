"""
Correct & Stable LSTM-based Encoder-Decoder for Machine Translation
with Attention, Bidirectional Encoder, proper hidden projection,
and correct masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Encoder
# -------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout=0.3, bidirectional=True):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embed(src))

        # pack padded
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        outputs, (h_n, c_n) = self.lstm(packed)

        # unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, (h_n, c_n)


# -------------------------
# Attention
# -------------------------
class Attention(nn.Module):
    """
    Dot-product attention compatible with bidirectional encoder.
    """
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.linear = nn.Linear(enc_dim + dec_dim, dec_dim)

    def forward(self, decoder_outputs, encoder_outputs, src_mask):
        """
        decoder_outputs: [B, T_dec, dec_dim]
        encoder_outputs: [B, T_src, enc_dim]
        src_mask: [B, 1, T_src]
        """
        # Attention scores
        scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
        scores = scores.masked_fill(~src_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights


# -------------------------
# Decoder
# -------------------------
class LSTMDecoder(nn.Module):
    """LSTM Decoder with Attention"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 enc_hidden_dim, dropout=0.3, attention=True):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_enabled = attention

        # encoder outputs are bidirectional → 2 * enc_hidden_dim
        self.enc_out_dim = enc_hidden_dim * 2

        # project encoder hidden to decoder hidden
        self.init_h = nn.Linear(self.enc_out_dim, hidden_dim)
        self.init_c = nn.Linear(self.enc_out_dim, hidden_dim)

        # NEW: project encoder outputs to decoder hidden dim
        self.encoder_proj = nn.Linear(self.enc_out_dim, hidden_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention: dot-product
        if attention:
            self.att_combine = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_input, encoder_outputs, encoder_hidden, src_lengths):
        B, T_src, _ = encoder_outputs.shape
        device = encoder_outputs.device

        # mask
        src_mask = torch.arange(T_src, device=device).unsqueeze(0) < src_lengths.unsqueeze(1)
        src_mask = src_mask.unsqueeze(1)

        # Retrieve bidirectional hidden state
        h_n, c_n = encoder_hidden
        h_fw = h_n[-2]
        h_bw = h_n[-1]
        c_fw = c_n[-2]
        c_bw = c_n[-1]

        h0 = torch.tanh(self.init_h(torch.cat([h_fw, h_bw], dim=-1)))
        c0 = torch.tanh(self.init_c(torch.cat([c_fw, c_bw], dim=-1)))
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)

        embedded = self.dropout(self.embed(tgt_input))
        dec_outputs, hidden = self.lstm(embedded, (h0, c0))

        if self.attention_enabled:

            # ⭐ project encoder outputs → dec hidden dim
            proj_encoder_outputs = self.encoder_proj(encoder_outputs)

            # attention scores
            scores = torch.bmm(dec_outputs, proj_encoder_outputs.transpose(1, 2))
            scores = scores.masked_fill(~src_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)

            # context vector
            context = torch.bmm(attn_weights, proj_encoder_outputs)

            # concatenate
            combined = torch.cat([dec_outputs, context], dim=-1)
            dec_outputs = torch.tanh(self.att_combine(combined))

        logits = self.out(dec_outputs)
        return logits
# -------------------------
# Seq2Seq Wrapper
# -------------------------
class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()

        self.encoder = LSTMEncoder(
            src_vocab_size, embed_dim, hidden_dim, num_layers,
            dropout=dropout, bidirectional=bidirectional
        )

        self.decoder = LSTMDecoder(
            tgt_vocab_size, embed_dim, hidden_dim, num_layers,
            enc_hidden_dim=hidden_dim,
            dropout=dropout,
            attention=True
        )

    def forward(self, src, tgt_input, src_lengths):
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        logits = self.decoder(tgt_input, encoder_outputs, encoder_hidden, src_lengths)
        return logits

    def encode(self, src, src_lengths):
        return self.encoder(src, src_lengths)