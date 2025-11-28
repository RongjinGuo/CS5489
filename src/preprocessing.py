"""
Preprocessing and tokenization for MT project
Supports both word-level and BPE tokenization
"""
import sentencepiece as spm
from pathlib import Path
from collections import Counter
import re

class WordTokenizer:
    """Word-level tokenizer"""
    
    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
    
    def build_vocab(self, sentences):
        """Build vocabulary from sentences"""
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            word_freq.update(words)
        
        # Add special tokens
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.special_tokens.items()}
        
        # Add most frequent words
        idx = len(self.special_tokens)
        for word, freq in word_freq.most_common(self.vocab_size - len(self.special_tokens)):
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Built vocabulary with {len(self.word2idx)} tokens")
        return self
    
    def encode(self, sentence, add_bos=False, add_eos=True):
        """Encode sentence to indices"""
        words = sentence.lower().split()
        indices = []
        
        if add_bos:
            indices.append(self.word2idx['<bos>'])
        
        for word in words:
            indices.append(self.word2idx.get(word, self.word2idx['<unk>']))
        
        if add_eos:
            indices.append(self.word2idx['<eos>'])
        
        return indices
    
    def decode(self, indices):
        """Decode indices to sentence"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<eos>']:
                break
            if idx not in [self.word2idx['<pad>'], self.word2idx['<bos>']]:
                words.append(self.idx2word.get(idx, '<unk>'))
        return ' '.join(words)
    def get_vocab_size(self):
        """For compatibility with BPETokenizer"""
        return len(self.word2idx)

class BPETokenizer:
    """BPE tokenizer using SentencePiece"""

    def __init__(self, model_path=None):
        self.sp = None
        self.model_path = model_path

        # If model exists → load it immediately
        if model_path and Path(model_path).exists():
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            print(f"[BPETokenizer] Loaded existing BPE model: {model_path}")

    def train(self, sentences, vocab_size=8000, model_prefix="models/bpe"):
        """
        Train BPE model only if it does NOT already exist.
        Otherwise load it directly.
        """
        model_file = f"{model_prefix}.model"

        # ================================
        # 1. If model already exists → skip training
        # ================================
        if Path(model_file).exists():
            print(f"[BPETokenizer] Found existing model: {model_file}, skip training.")
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_file)
            self.model_path = model_file
            return self

        # ================================
        # 2. Train model (first time only)
        # ================================
        print(f"[BPETokenizer] Training new BPE model → {model_file}")
        Path("models").mkdir(exist_ok=True)

        temp_file = "models/bpe_training_corpus.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for line in sentences:
                f.write(line.strip() + "\n")

        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>"
        )

        # Load trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.model_path = model_file

        print(f"[BPETokenizer] Training finished. Model saved → {model_file}")

        # Clean temp files
        Path(temp_file).unlink(missing_ok=True)
        return self

    def encode(self, sentence, add_bos=False, add_eos=True):
        if self.sp is None:
            raise ValueError("BPE model not loaded.")
        return self.sp.encode(sentence, out_type=int, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids):
        if self.sp is None:
            raise ValueError("BPE model not loaded.")
        return self.sp.decode(ids)
    
    def load(self, model_path):
        """Explicitly load an existing SentencePiece model."""
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"[BPETokenizer] Loaded model from {model_path}")
        return self
        
    def get_vocab_size(self):
        return len(self.sp) if self.sp else 0



class MTDataset:
    """Dataset class for machine translation"""
    
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_length=128):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
        # Load sentences
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f]
        
        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f]
        
        assert len(self.src_sentences) == len(self.tgt_sentences), "Mismatched data"
        print(f"Loaded {len(self.src_sentences)} sentence pairs")
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sent = self.src_sentences[idx]
        tgt_sent = self.tgt_sentences[idx]
        
        # Encode
        src_ids = self.src_tokenizer.encode(src_sent, add_bos=False, add_eos=True)
        tgt_input_ids = self.tgt_tokenizer.encode(tgt_sent, add_bos=True, add_eos=False)
        tgt_output_ids = self.tgt_tokenizer.encode(tgt_sent, add_bos=False, add_eos=True)
        
        # Truncate to max_length
        src_ids = src_ids[:self.max_length]
        # Ensure tgt_input and tgt_output have the same length for training
        # tgt_input: [BOS, w1, w2, ..., wn] (length n+1)
        # tgt_output: [w1, w2, ..., wn, EOS] (length n+1)
        # They should have the same length
        max_tgt_len = min(len(tgt_input_ids), len(tgt_output_ids), self.max_length)
        tgt_input_ids = tgt_input_ids[:max_tgt_len]
        tgt_output_ids = tgt_output_ids[:max_tgt_len]
        
        return {
            'src': src_ids,
            'tgt_input': tgt_input_ids,
            'tgt_output': tgt_output_ids,
            'src_text': src_sent,
            'tgt_text': tgt_sent
        }


def collate_fn(batch, pad_id=0):
    """Collate function for DataLoader"""
    src_batch = [item['src'] for item in batch]
    tgt_input_batch = [item['tgt_input'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]
    
    # Pad sequences
    src_lengths = [len(s) for s in src_batch]
    tgt_input_lengths = [len(t) for t in tgt_input_batch]
    tgt_output_lengths = [len(t) for t in tgt_output_batch]
    
    max_src_len = max(src_lengths) if src_lengths else 1
    max_tgt_input_len = max(tgt_input_lengths) if tgt_input_lengths else 1
    max_tgt_output_len = max(tgt_output_lengths) if tgt_output_lengths else 1
    
    src_padded = []
    tgt_input_padded = []
    tgt_output_padded = []
    
    for i in range(len(batch)):
        src_padded.append(src_batch[i] + [pad_id] * (max_src_len - len(src_batch[i])))
        tgt_input_padded.append(tgt_input_batch[i] + [pad_id] * (max_tgt_input_len - len(tgt_input_batch[i])))
        tgt_output_padded.append(tgt_output_batch[i] + [pad_id] * (max_tgt_output_len - len(tgt_output_batch[i])))
    
    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_input_lengths  # Use tgt_input_lengths for decoder
    }

