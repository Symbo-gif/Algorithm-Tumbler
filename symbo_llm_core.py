# Copyright 2025 Damien Davison, Michael Maillet, and Sacha Davison, Recursive AI Devs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Symbo LLM Core - Neural Language Model for SYMBO_AGENTIC_REASONERS
================================================

Trainable transformer-based language model for mathematical reasoning.
Supports:
- Real training with backpropagation
- Persistent model checkpoints
- Continuous learning from verified solutions
- Knowledge base integration for RAG

Adapted for SYMBO_AGENTIC_REASONERS Phase 5 integration.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Optional torch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    # ImportError: torch not installed
    # RuntimeError: torch version incompatible with Python version (e.g., Python 3.14)
    # OSError: library loading issues
    TORCH_AVAILABLE = False
    # Create stub classes for when torch isn't available
    class nn:
        class Module:
            pass


class SymboLLMCore(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer-based language model for mathematical reasoning.

    Architecture:
    - Token embedding layer
    - Positional encoding
    - Multi-head attention layers
    - Feed-forward layers
    - Output projection

    Designed to be trainable and persistent.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        if not TORCH_AVAILABLE:
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.max_seq_len = max_seq_len
            self.device = "cpu"
            self.knowledge_store: Dict[str, Dict[str, Any]] = {}
            self.training_examples = 0
            self.training_epochs = 0
            self.last_loss = 0.0
            return

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = device if torch.cuda.is_available() else "cpu"

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Knowledge base for retrieval-augmented generation
        self.knowledge_store: Dict[str, Dict[str, Any]] = {}

        # Training stats
        self.training_examples = 0
        self.training_epochs = 0
        self.last_loss = 0.0

        self.to(self.device)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass"""
        if not TORCH_AVAILABLE:
            return None

        batch_size, seq_len = x.shape

        # Embed tokens
        x = self.token_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Dropout
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.2
    ) -> List[int]:
        """Generate tokens given a prompt"""
        if not TORCH_AVAILABLE:
            return prompt_tokens

        self.eval()

        PAD_ID = 0
        BOS_ID = 2
        EOS_ID = 3

        generated = prompt_tokens.copy()
        consecutive_eos = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_seq = torch.tensor([generated[-self.max_seq_len:]],
                                        dtype=torch.long, device=self.device)

                logits = self(input_seq)
                next_token_logits = logits[0, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for prev_token in set(generated[-50:]):
                        next_token_logits[prev_token] /= repetition_penalty

                # Penalize special tokens
                next_token_logits[PAD_ID] = float('-inf')
                next_token_logits[BOS_ID] = float('-inf')
                next_token_logits[EOS_ID] -= 5.0

                # Apply top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    indices_to_remove = next_token_logits < top_k_vals[-1]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token == EOS_ID:
                    consecutive_eos += 1
                    if consecutive_eos >= 2:
                        break
                else:
                    consecutive_eos = 0

                generated.append(next_token)

                if next_token == PAD_ID:
                    break

        return generated

    def compute_loss(
        self,
        input_ids: 'torch.Tensor',
        target_ids: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """Compute cross-entropy loss"""
        if not TORCH_AVAILABLE:
            return 0.0

        logits = self(input_ids)
        logits = logits.view(-1, self.vocab_size)
        target_ids = target_ids.view(-1)
        loss = F.cross_entropy(logits, target_ids, ignore_index=0)
        return loss

    def add_to_knowledge_store(self, key: str, data: Dict[str, Any]):
        """Add entry to knowledge store"""
        self.knowledge_store[key] = {
            **data,
            'added_at': datetime.now().isoformat()
        }

    def query_knowledge_store(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query knowledge store with fuzzy matching"""
        query_lower = query.lower().strip()
        query_normalized = query_lower.replace('?', '').replace('.', '').replace('!', '').strip()

        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
                      'does', 'do', 'can', 'could', 'would', 'should', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'to', 'of',
                      'in', 'for', 'on', 'with', 'at', 'by', 'from', 'it', 'this', 'that'}

        query_words = set(
            word.lower().strip('?.,!')
            for word in query.split()
            if len(word) > 1 and word.lower() not in stop_words
        )

        if not query_words:
            query_words = {query_lower}

        results = []
        for key, data in self.knowledge_store.items():
            score = 0

            if 'prompt' not in data:
                continue

            prompt_lower = data['prompt'].lower().strip()
            prompt_normalized = prompt_lower.replace('?', '').replace('.', '').replace('!', '').strip()

            if query_normalized == prompt_normalized:
                score = 1000
            elif query_lower in prompt_lower or prompt_lower in query_lower:
                score = 500

            if score < 500:
                prompt_words = set(
                    word.strip('?.,!').lower()
                    for word in prompt_lower.split()
                    if len(word) > 1
                )

                significant_overlap = query_words & prompt_words
                score += len(significant_overlap) * 15

                query_nums = set(w for w in query_words if w.isdigit())
                prompt_nums = set(w for w in prompt_words if w.isdigit())
                if query_nums and query_nums == prompt_nums:
                    score += 50

                len_ratio = len(prompt_lower) / max(len(query_lower), 1)
                if len_ratio < 0.3 or len_ratio > 3.0:
                    score = int(score * 0.5)

            if 'category' in data:
                cat_lower = data['category'].lower()
                for qw in query_words:
                    if qw in cat_lower:
                        score += 5

            if score > 0:
                results.append((score, data))

        results.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in results[:top_k]]

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        if not TORCH_AVAILABLE:
            # Save only knowledge store
            with open(filepath + '.json', 'w') as f:
                json.dump({
                    'knowledge_store': self.knowledge_store,
                    'training_examples': self.training_examples,
                    'training_epochs': self.training_epochs,
                }, f)
            return

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'max_seq_len': self.max_seq_len,
            'knowledge_store': self.knowledge_store,
            'training_examples': self.training_examples,
            'training_epochs': self.training_epochs,
            'last_loss': self.last_loss,
            'timestamp': datetime.now().isoformat()
        }

        temp_filepath = filepath + '.tmp'
        try:
            torch.save(checkpoint, temp_filepath)
            if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 1000:
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
        except Exception as e:
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError:
                    pass  # Ignore errors during temp file cleanup

    def load_checkpoint(self, filepath: str) -> bool:
        """Load model checkpoint"""
        if not TORCH_AVAILABLE:
            json_path = filepath + '.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.knowledge_store = data.get('knowledge_store', {})
                    self.training_examples = data.get('training_examples', 0)
                    self.training_epochs = data.get('training_epochs', 0)
                return True
            return False

        if not os.path.exists(filepath):
            return False

        # SECURITY: Use weights_only=True to prevent arbitrary code execution
        # via malicious pickle payloads in checkpoint files
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.knowledge_store = checkpoint.get('knowledge_store', {})
        self.training_examples = checkpoint.get('training_examples', 0)
        self.training_epochs = checkpoint.get('training_epochs', 0)
        self.last_loss = checkpoint.get('last_loss', 0.0)

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        stats = {
            'training_examples': self.training_examples,
            'training_epochs': self.training_epochs,
            'last_loss': self.last_loss,
            'knowledge_entries': len(self.knowledge_store),
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'device': str(self.device),
            'torch_available': TORCH_AVAILABLE,
        }

        if TORCH_AVAILABLE and hasattr(self, 'parameters'):
            stats['parameters'] = sum(p.numel() for p in self.parameters())

        return stats


class TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """Single transformer block with self-attention and feed-forward"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        if not TORCH_AVAILABLE:
            return

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass with residual connections"""
        if not TORCH_AVAILABLE:
            return x

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class SimpleTokenizer:
    """
    Character-level tokenizer with mathematical symbol support.
    """

    MATH_SYMBOLS = {
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ',
        'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ',
        'υ', 'φ', 'χ', 'ψ', 'ω',
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ',
        'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ',
        'Φ', 'Χ', 'Ψ', 'Ω',
        '∀', '∃', '∄', '∴', '∵', '¬', '∧', '∨', '⊕', '⊗',
        '→', '←', '↔', '⇒', '⇐', '⇔', '⊢', '⊨', '⊤', '⊥',
        '∈', '∉', '∋', '∌', '∅', '⊂', '⊃', '⊄', '⊅', '⊆',
        '⊇', '⊈', '⊉', '∪', '∩', '∖', '∆', '⊎',
        'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ', 'ℍ', 'ℙ', 'ℵ', 'ℶ',
        '∫', '∬', '∭', '∮', '∯', '∰', '∂', '∇',
        '∑', '∏', '∐', '√', '∛', '∜',
        '≤', '≥', '≠', '≈', '≅', '≡', '≢', '∝', '≪', '≫',
        '±', '∓', '×', '÷', '·', '∘', '⊙', '⊛',
        '∞', '°', '′', '″', '‴',
    }

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        self.char_to_id = {**self.special_tokens}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        next_id = len(self.special_tokens)

        # ASCII printable
        for i in range(32, 127):
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                self.id_to_char[next_id] = char
                next_id += 1

        # Extended Latin
        for i in range(128, 256):
            try:
                char = chr(i)
                if char not in self.char_to_id and char.isprintable():
                    self.char_to_id[char] = next_id
                    self.id_to_char[next_id] = char
                    next_id += 1
            except (ValueError, UnicodeError):
                continue

        # Math symbols
        for symbol in sorted(self.MATH_SYMBOLS):
            if symbol not in self.char_to_id:
                self.char_to_id[symbol] = next_id
                self.id_to_char[next_id] = symbol
                next_id += 1

        self.actual_vocab_size = next_id

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = [self.special_tokens['<BOS>']]
        for char in text:
            token_id = self.char_to_id.get(char, self.special_tokens['<UNK>'])
            tokens.append(token_id)
        tokens.append(self.special_tokens['<EOS>'])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        chars = []
        for token_id in tokens:
            if token_id in [0, 2, 3]:
                continue
            char = self.id_to_char.get(token_id, '?')
            chars.append(char)
        return ''.join(chars)

    def save(self, filepath: str):
        """Save tokenizer vocabulary"""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': {int(k): v for k, v in self.id_to_char.items()},
            'vocab_size': self.vocab_size,
            'actual_vocab_size': getattr(self, 'actual_vocab_size', len(self.char_to_id))
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Load tokenizer vocabulary"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self.vocab_size = data.get('vocab_size', 10000)
        self.actual_vocab_size = data.get('actual_vocab_size', len(self.char_to_id))
