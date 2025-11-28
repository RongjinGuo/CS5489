"""
GRU-based Encoder-Decoder for Machine Translation
(with bidirectional encoder + attention, dimension-safe)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Encoder
# -------------------------
class GRUEncoder(nn.Module):
    """GRU Encoder"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths):
        """
        src: [B, T_src]
        src_lengths: [B]
        returns:
            outputs: [B, T_src, hidden_dim * num_directions]
            hidden: [num_layers * num_directions, B, hidden_dim]
        """
        embedded = self.dropout(self.embedding(src))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden


# -------------------------
# Decoder
# -------------------------
class GRUDecoder(nn.Module):
    """GRU Decoder with dot-product attention"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout, encoder_hidden_dim, attention=True, bidirectional_encoder=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_enabled = attention

        self.enc_hidden_dim = encoder_hidden_dim
        self.enc_directions = 2 if bidirectional_encoder else 1
        self.enc_out_dim = self.enc_hidden_dim * self.enc_directions  # 1024 if 512 & bidirectional

        # 用于把 encoder 最后一层 (fw+bw) 初始化到 decoder hidden 维度
        self.encoder_hidden_proj = nn.Linear(self.enc_out_dim, hidden_dim)

        # 用于把 encoder 的输出特征从 1024 映射到 512，方便做 dot attention
        self.encoder_output_proj = nn.Linear(self.enc_out_dim, hidden_dim)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        if attention:
            # 输入是 [dec_output(512) ; context(512)] → 1024
            self.attention_layer = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
            # 再跟 embedding 拼一下
            self.attention_combine = nn.Linear(hidden_dim + embed_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_outputs, encoder_hidden, src_lengths):
        """
        tgt: [B, T_tgt]
        encoder_outputs: [B, T_src, enc_out_dim]
        encoder_hidden: [num_layers * num_directions, B, enc_hidden_dim]
        src_lengths: [B]
        """
        device = encoder_outputs.device
        B, T_src, _ = encoder_outputs.shape

        # ---- 1. 初始化 decoder hidden ----
        # encoder_hidden: [L*D, B, H]
        # 我们取最后一层的 fw & bw：
        if self.enc_directions == 2:
            h_fw = encoder_hidden[-2]  # [B, H]
            h_bw = encoder_hidden[-1]  # [B, H]
            h_enc_cat = torch.cat([h_fw, h_bw], dim=-1)  # [B, 2H]
        else:
            h_enc_cat = encoder_hidden[-1]  # [B, H]

        # 投影到 decoder hidden 维度
        h0 = torch.tanh(self.encoder_hidden_proj(h_enc_cat))  # [B, hidden_dim]
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)    # [num_layers, B, hidden_dim]
        hidden = h0

        # ---- 2. 准备 decoder 输入嵌入 ----
        embedded = self.dropout(self.embedding(tgt))  # [B, T_tgt, embed_dim]

        # ---- 3. 运行 decoder GRU ----
        dec_outputs, hidden = self.gru(embedded, hidden)  # dec_outputs: [B, T_tgt, hidden_dim]
        dec_outputs = self.dropout(dec_outputs)

        # ---- 4. Attention（如果启用）----
        if self.attention_enabled:
            # 投影 encoder_outputs 到 decoder hidden 维度
            enc_proj = self.encoder_output_proj(encoder_outputs)  # [B, T_src, hidden_dim]

            # mask: True 表示 valid
            src_mask = torch.arange(T_src, device=device).unsqueeze(0) < src_lengths.unsqueeze(1)
            src_mask = src_mask.unsqueeze(1)  # [B, 1, T_src]

            # scores: [B, T_tgt, T_src]
            scores = torch.bmm(dec_outputs, enc_proj.transpose(1, 2))
            scores = scores.masked_fill(~src_mask, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)           # [B, T_tgt, T_src]
            context = torch.bmm(attn_weights, enc_proj)        # [B, T_tgt, hidden_dim]

            # concat dec_output 和 context
            combined = torch.cat([dec_outputs, context], dim=-1)   # [B, T_tgt, 2*hidden_dim]
            attended = torch.tanh(self.attention_layer(combined))  # [B, T_tgt, hidden_dim]

            # 再和 embedding 拼一下
            out_combined = torch.cat([attended, embedded], dim=-1)  # [B, T_tgt, hidden_dim+embed_dim]
            dec_outputs = torch.tanh(self.attention_combine(out_combined))  # [B, T_tgt, hidden_dim]

        # ---- 5. 输出到词表 ----
        logits = self.output_proj(dec_outputs)  # [B, T_tgt, vocab_size]
        return logits


# -------------------------
# Seq2Seq Wrapper
# -------------------------
class GRUSeq2Seq(nn.Module):
    """Complete GRU Seq2Seq Model"""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256,
                 hidden_dim=512, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()

        self.encoder = GRUEncoder(
            src_vocab_size, embed_dim, hidden_dim,
            num_layers, dropout, bidirectional=bidirectional
        )

        self.decoder = GRUDecoder(
            tgt_vocab_size, embed_dim, hidden_dim,
            num_layers, dropout,
            encoder_hidden_dim=hidden_dim,
            attention=True,
            bidirectional_encoder=bidirectional
        )

    def forward(self, src, tgt_input, src_lengths):
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        logits = self.decoder(tgt_input, encoder_outputs, encoder_hidden, src_lengths)
        return logits

    def encode(self, src, src_lengths):
        return self.encoder(src, src_lengths)