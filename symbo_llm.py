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
Symbo LLM Adapter - Mathematical Knowledge Store and Learning System
====================================================================

Provides the LLM interface for mathematical problem solving with:
- Neural network training (when PyTorch available)
- Persistent knowledge store (294K+ entries across 60 domains)
- Continuous learning from exploration discoveries
- Batch learning from training datasets (3000+ problems)
- Knowledge retention auditing (95%+ target)
- Gatekeeper verification for quality control
- Learning Enhancement Team integration (6 specialists)
- Fallback to symbolic methods when neural unavailable

Key Methods:
- learn_batch(): Bulk training from domain-specific datasets
- learn_batch_with_gatekeeper(): Training with verification
- query_exact(): Direct knowledge lookup by problem text
- get_knowledge_count(): Current knowledge store size
- get_stats(): Model and training statistics

Knowledge Store Structure:
- Keys: Normalized problem text strings
- Values: Dict with 'prompt', 'response', 'category', 'problem_id', 'timestamp', 'added_at'

Training Data Location: data/training/<domain>_training.json
Checkpoint Location: data/symbo_llm/unified/symbo_llm_latest.pt
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging
import os
import re
import random
from datetime import datetime

# Optional torch import
try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    # ImportError: torch not installed
    # RuntimeError: torch version incompatible with Python version (e.g., Python 3.14)
    # OSError: library loading issues
    TORCH_AVAILABLE = False

# Import from local package
from .symbo_llm_core import SymboLLMCore, SimpleTokenizer, TORCH_AVAILABLE as CORE_TORCH_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class LLMTask:
    """
    Task specification for Symbo LLM

    Attributes:
        prompt: Mathematical query or problem description
        context: Optional context (problem state, history, etc.)
        task_type: Type of task (reasoning, proof, computation, simplify)
        max_tokens: Maximum response length
        temperature: Randomness (0=deterministic, 1=creative)
    """
    prompt: str
    context: Optional[str] = None
    task_type: str = "reasoning"  # reasoning, proof, computation, simplify
    max_tokens: int = 512
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymboLLMAdapter:
    """
    LLM Adapter for SYMBO_AGENTIC_REASONERS Mathematical Problem Solving.

    The central knowledge management system for mathematical learning. Maintains
    a persistent knowledge store that grows through training and exploration.

    Features:
    - Persistent knowledge store with 294K+ entries across 60 mathematical domains
    - Batch learning from domain-specific training datasets (learn_batch)
    - Quality-verified learning with Gatekeeper (learn_batch_with_gatekeeper)
    - Exact query lookup for problem->answer retrieval (query_exact)
    - Learning Enhancement Team integration (6 specialists for advanced learning)
    - Neural network training when PyTorch is available
    - Symbolic fallback mode when neural unavailable
    - Automatic checkpoint persistence to data/symbo_llm/unified/

    Usage:
        llm = SymboLLMAdapter(checkpoint_dir='data/symbo_llm/unified')

        # Training
        llm.learn_batch(problems)  # Bulk training
        llm.learn_from_interaction(problem, solution)  # Single problem

        # Querying
        answer = llm.query_exact("solve(x^2 - 4)")
        count = llm.get_knowledge_count()
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        device: str = "cuda",
        checkpoint_dir: str = None
    ):
        """
        Initialize LLM adapter

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory for saving/loading checkpoints
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.logger = logging.getLogger("symbo_agentic_reasoners.symbo.llm")

        # Determine device
        if TORCH_AVAILABLE:
            self.device = device if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Initializing Symbo LLM on {self.device}")
        else:
            self.device = "cpu"
            self.logger.info("PyTorch not available - using symbolic fallback mode")

        # Set checkpoint directory (default: data/symbo_llm/unified for persistence)
        # All training automatically merges into unified checkpoint
        if checkpoint_dir is None:
            checkpoint_dir = "data/symbo_llm/unified"

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "symbo_llm_latest.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
            self.logger.info("Tokenizer loaded from checkpoint")

        # Initialize model
        self.model = SymboLLMCore(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=3,
            ff_dim=512,
            max_seq_len=512,
            dropout=0.1,
            device=self.device
        )

        # Try to load existing checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "symbo_llm_latest.pt")
        if os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path)
            self.logger.info("Loaded existing model checkpoint")
        else:
            self.logger.info("No checkpoint found - using fresh model")

        # Optimizer for continuous learning
        if TORCH_AVAILABLE:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)
        else:
            self.optimizer = None

        # Statistics
        self.total_queries = 0
        self.successful_generations = 0

        # Learning Enhancement Team (lazy-loaded)
        self._learning_supervisor = None
        self._enhancement_enabled = False

        # Initialize mathematical response patterns
        self._init_math_patterns()

        self.logger.info("Symbo LLM initialization complete!")
        self.logger.info(f"Model stats: {self.model.get_stats()}")

    def _init_math_patterns(self):
        """Initialize mathematical response patterns for fallback"""
        # Mathematical operation patterns
        self.math_operations = {
            'triggers': [
                'differentiate', 'derivative', 'diff', 'd/dx',
                'integrate', 'integral', 'antiderivative',
                'solve', 'find x', 'find the value',
                'simplify', 'reduce', 'factor',
                'expand', 'distribute',
                'limit', 'lim',
                'sum', 'series', 'sequence',
            ],
            'responses': [
                "This requires applying {operation} to the expression.",
                "For {operation}, we follow the standard rules.",
                "The {operation} can be computed step by step.",
            ]
        }

        # Proof patterns
        self.proof_patterns = {
            'triggers': [
                'prove', 'proof', 'show that', 'demonstrate',
                'verify', 'confirm', 'establish',
            ],
            'responses': [
                "To prove this, we proceed by {method}.",
                "The proof follows from {principle}.",
                "We can establish this using {technique}.",
            ]
        }

        # Definition patterns
        self.definition_patterns = {
            'triggers': [
                'what is', 'define', 'meaning of', 'explain',
            ],
            'responses': [
                "In mathematics, {term} refers to...",
                "The definition of {term} is...",
                "{term} is a mathematical concept that...",
            ]
        }

    def _check_math_patterns(self, prompt: str) -> Optional[str]:
        """Check if prompt matches mathematical patterns"""
        prompt_lower = prompt.lower().strip()

        # Check operation patterns
        for trigger in self.math_operations['triggers']:
            if trigger in prompt_lower:
                response = random.choice(self.math_operations['responses'])
                return response.format(operation=trigger)

        # Check proof patterns
        for trigger in self.proof_patterns['triggers']:
            if trigger in prompt_lower:
                methods = ['induction', 'contradiction', 'direct proof', 'construction']
                response = random.choice(self.proof_patterns['responses'])
                return response.format(
                    method=random.choice(methods),
                    principle='fundamental axioms',
                    technique='established theorems'
                )

        # Check definition patterns
        for trigger in self.definition_patterns['triggers']:
            if trigger in prompt_lower:
                # Extract potential term
                term = prompt_lower.replace(trigger, '').strip().strip('?.')
                response = random.choice(self.definition_patterns['responses'])
                return response.format(term=term if term else 'this concept')

        return None

    def _is_gibberish(self, text: str) -> bool:
        """
        Detect if generated text is gibberish/corrupted output

        Returns True if the text appears to be nonsensical
        """
        if not text or len(text) < 5:
            return True

        # Check 1: Too many repeated characters (more than 3 in a row)
        if re.search(r'(.)\1{3,}', text):
            return True

        # Check 2: Very low vowel ratio (natural English needs vowels)
        vowels = sum(1 for c in text.lower() if c in 'aeiou')
        letters = sum(1 for c in text if c.isalpha())
        if letters > 10 and vowels / max(letters, 1) < 0.15:
            return True

        # Check 3: Very low space ratio (natural language has word breaks)
        spaces = text.count(' ')
        if len(text) > 20 and spaces / len(text) < 0.05:
            return True

        # Check 4: Too many non-alphanumeric characters
        non_alphanum = sum(1 for c in text if not c.isalnum() and c not in ' .,!?\'"-:;()[]{}^_+=')
        if len(text) > 10 and non_alphanum / len(text) > 0.3:
            return True

        # Check 5: Consecutive consonant runs too long (>6 consonants in a row)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{7,}', text.lower()):
            return True

        # Check 6: Check for some recognizable common words
        common_words = {
            'the', 'a', 'is', 'in', 'to', 'and', 'of', 'it', 'for', 'on',
            'are', 'with', 'as', 'be', 'this', 'have', 'from', 'by', 'or',
            # Math-specific words
            'let', 'if', 'then', 'where', 'such', 'that', 'given', 'find',
            'solve', 'prove', 'show', 'equals', 'plus', 'minus', 'times',
        }
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        common_found = len(words & common_words)
        if len(words) > 5 and common_found == 0:
            return True

        return False

    def _generate_contextual_response(self, prompt: str, context: str = None) -> str:
        """Generate a contextual response for mathematical queries"""
        prompt_lower = prompt.lower()

        # Calculation request
        if any(w in prompt_lower for w in ['calculate', 'compute', 'evaluate', 'find']):
            return "This calculation requires systematic evaluation. Let me work through the steps."

        # Proof request
        if any(w in prompt_lower for w in ['prove', 'show', 'demonstrate', 'verify']):
            return "This proof requires careful logical reasoning. The approach depends on the specific structure."

        # Simplification request
        if any(w in prompt_lower for w in ['simplify', 'reduce', 'factor']):
            return "To simplify this expression, we apply algebraic transformations systematically."

        # Question with mathematical content
        if '?' in prompt or any(w in prompt_lower for w in ['what', 'how', 'why']):
            return "That's a mathematical question that requires analysis. Could you provide more specific context?"

        # Generic mathematical inquiry
        return "I understand this as a mathematical problem. Let me analyze the structure and approach."

    def handle_task(self, task: LLMTask) -> str:
        """
        Process LLM task with knowledge retrieval and fallback

        Args:
            task: LLMTask specification

        Returns:
            Generated response string
        """
        self.total_queries += 1

        try:
            # First: check knowledge base with fuzzy matching
            kb_results = self.model.query_knowledge_store(task.prompt, top_k=3)

            if kb_results and len(kb_results) > 0:
                # Found relevant knowledge - pick best match with valid response
                for result in kb_results:
                    if 'response' in result:
                        response = result['response']
                        # Skip empty responses
                        if not response or len(response.strip()) == 0:
                            continue
                        # Skip corrupted/invalid responses
                        if '<UNK>' in response:
                            continue
                        # Skip placeholder responses
                        if 'still learning' in response.lower():
                            continue
                        # Accept short equation-style answers (e.g., "2", "pi^2/6")
                        # and longer explanatory responses
                        self.logger.debug(f"Knowledge base hit for: {task.prompt[:50]}")
                        self.successful_generations += 1
                        return response

            # Second: check mathematical patterns
            pattern_response = self._check_math_patterns(task.prompt)
            if pattern_response:
                self.successful_generations += 1
                return pattern_response

            # Third: try neural generation (if available)
            if TORCH_AVAILABLE:
                response = self._generate_response(task)
                if response and len(response.strip()) > 10:
                    self.successful_generations += 1
                    return response

            # Fourth: generate contextual response
            return self._generate_contextual_response(task.prompt, task.context)

        except Exception as e:
            self.logger.error(f"LLM task failed: {e}", exc_info=True)
            return "An error occurred processing that mathematical query. Please try rephrasing."

    def _generate_response(self, task: LLMTask) -> str:
        """
        Generate response using the neural model

        Args:
            task: LLMTask specification

        Returns:
            Generated text
        """
        if not TORCH_AVAILABLE:
            return ""

        # Build prompt in Q/A format (matches training format)
        prompt_text = f"Q: {task.prompt}\nA:"
        if task.context:
            prompt_text = f"Context: {task.context}\n\n{prompt_text}"

        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt_text)
        prompt_len = len(prompt_tokens)

        # Generate with repetition penalty
        generated_tokens = self.model.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=task.max_tokens,
            temperature=task.temperature,
            top_k=50,
            repetition_penalty=1.3
        )

        # Only decode the NEW tokens (after prompt)
        new_tokens = generated_tokens[prompt_len:]

        # Remove special tokens from output
        filtered_tokens = [t for t in new_tokens if t not in (0, 1, 2, 3)]

        if not filtered_tokens:
            return ""

        # Decode only the generated response
        response = self.tokenizer.decode(filtered_tokens)

        # Clean up whitespace
        response = response.strip()

        # Remove any repeated characters (artifact of char-level model)
        response = re.sub(r'(.)\1{4,}', r'\1\1', response)  # Max 2 repeated chars

        # Check if response is gibberish - reject if so
        if self._is_gibberish(response):
            self.logger.debug(f"Neural output rejected as gibberish: {response[:50]}...")
            return ""  # Return empty to trigger fallback

        return response

    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """
        Generate text response (Standard LLM interface)

        Args:
            prompt: Input text
            max_length: Max tokens
            temperature: Creativity

        Returns:
            Generated string
        """
        task = LLMTask(
            prompt=prompt,
            max_tokens=max_length,
            temperature=temperature,
            task_type="general"
        )
        return self.handle_task(task)

    def train(self, dataset: List[Dict[str, Any]], epochs: int = 50, batch_size: int = 4):
        """
        Train on mathematical examples with backpropagation

        Args:
            dataset: List of training examples
                     Each: {'prompt': str, 'response': str, 'category': str}
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("Training requires PyTorch - storing in knowledge base only")
            for item in dataset:
                if 'prompt' in item and 'response' in item:
                    key = f"{item.get('category', 'general')}_{len(self.model.knowledge_store)}"
                    self.model.add_to_knowledge_store(key, item)
            return

        self.logger.info(f"Starting training on {len(dataset)} examples for {epochs} epochs")

        self.model.train()

        # Prepare training data
        training_pairs = []
        for item in dataset:
            if 'prompt' in item and 'response' in item:
                prompt = item['prompt']
                response = item['response']

                # Store in knowledge base
                key = f"{item.get('category', 'general')}_{len(self.model.knowledge_store)}"
                self.model.add_to_knowledge_store(key, item)

                # Create training pair (prompt -> response)
                full_text = f"Q: {prompt}\nA: {response}"
                training_pairs.append(full_text)

        if not training_pairs:
            self.logger.warning("No training pairs found in dataset")
            return

        # Training loop
        total_loss = 0.0
        step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle data
            random.shuffle(training_pairs)

            # Process in batches
            for i in range(0, len(training_pairs), batch_size):
                batch_texts = training_pairs[i:i + batch_size]

                # Tokenize batch
                batch_tokens = [self.tokenizer.encode(text) for text in batch_texts]

                # Pad to same length
                max_len = max(len(tokens) for tokens in batch_tokens)
                max_len = min(max_len, self.model.max_seq_len)

                padded_tokens = []
                for tokens in batch_tokens:
                    if len(tokens) > max_len:
                        tokens = tokens[:max_len]
                    else:
                        tokens = tokens + [0] * (max_len - len(tokens))
                    padded_tokens.append(tokens)

                # Convert to tensors
                input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=self.device)

                # Target is same as input (language modeling)
                target_ids = input_ids.clone()

                # Forward pass
                loss = self.model.compute_loss(input_ids, target_ids)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                step += 1

            avg_epoch_loss = epoch_loss / max(len(training_pairs) / batch_size, 1)
            total_loss += avg_epoch_loss

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

        # Update model stats
        self.model.training_examples += len(dataset)
        self.model.training_epochs += epochs
        self.model.last_loss = total_loss / max(epochs, 1)

        # Save checkpoint
        self.save()

        self.logger.info(f"Training complete! Final loss: {self.model.last_loss:.4f}")
        self.logger.info(f"Total examples trained: {self.model.training_examples}")

    def learn_from_interaction(self, user_input: str, response: str, category: str = "interaction"):
        """
        Continuously learn from a problem-solving interaction

        This performs a single gradient update based on the interaction.

        Args:
            user_input: User's mathematical query
            response: System's response/solution
            category: Category for the interaction
        """
        # Add to knowledge store with both indexed key and normalized prompt key
        # The indexed key preserves all metadata
        indexed_key = f"{category}_{len(self.model.knowledge_store)}"
        entry_data = {
            'prompt': user_input,
            'response': response,
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
        self.model.add_to_knowledge_store(indexed_key, entry_data)

        # Also add with normalized prompt as key for fast lookup by solver
        # This enables O(1) lookup when solver queries with the same problem text
        normalized_prompt = self._normalize_for_lookup(user_input)
        self.model.add_to_knowledge_store(normalized_prompt, entry_data)

        if not TORCH_AVAILABLE:
            return

        # Perform a single training step
        self.model.train()

        try:
            # Create training text
            full_text = f"Q: {user_input}\nA: {response}"
            tokens = self.tokenizer.encode(full_text)

            # Truncate if needed
            if len(tokens) > self.model.max_seq_len:
                tokens = tokens[:self.model.max_seq_len]

            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            target_ids = input_ids.clone()

            # Single gradient step
            loss = self.model.compute_loss(input_ids, target_ids)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.logger.debug(f"Learned from interaction - Loss: {loss.item():.4f}")

            # Periodically save (every 10 interactions)
            if len(self.model.knowledge_store) % 10 == 0:
                self.save()

        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {e}")

    def save(self, filepath: str = None):
        """
        Save model checkpoint

        Args:
            filepath: Optional custom save path

        Returns:
            True if successful
        """
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "symbo_llm_latest.pt")

        try:
            self.model.save_checkpoint(filepath)

            # Also save tokenizer
            tokenizer_path = os.path.join(self.checkpoint_dir, "tokenizer.json")
            self.tokenizer.save(tokenizer_path)

            self.logger.info(f"Saved checkpoint to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False

    def save_checkpoint(self, filepath: str = None):
        """Alias for save() for API compatibility."""
        return self.save(filepath)

    @property
    def knowledge_store(self) -> Dict[str, Any]:
        """Access to the underlying knowledge store for direct lookups."""
        return self.model.knowledge_store

    def load(self, filepath: str = None):
        """
        Load model checkpoint

        Args:
            filepath: Optional custom load path

        Returns:
            True if successful
        """
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, "symbo_llm_latest.pt")

        try:
            success = self.model.load_checkpoint(filepath)

            if success:
                # Also load tokenizer
                tokenizer_path = os.path.join(self.checkpoint_dir, "tokenizer.json")
                if os.path.exists(tokenizer_path):
                    self.tokenizer.load(tokenizer_path)

                self.logger.info(f"Loaded checkpoint from {filepath}")

            return success
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def merge_from_checkpoint(self, checkpoint_path: str, deduplicate: bool = True) -> Dict[str, Any]:
        """
        Merge knowledge from another checkpoint into the current model.

        This allows combining knowledge from multiple training runs into
        the unified model. Knowledge is merged with optional deduplication.

        Args:
            checkpoint_path: Path to checkpoint file to merge from
            deduplicate: If True, skip entries with duplicate queries

        Returns:
            Dict with merge statistics
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint not found: {checkpoint_path}")
            return {'success': False, 'error': 'Checkpoint not found'}

        try:
            # Load the source checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            except TypeError:
                msg = (
                    "PyTorch version does not support safe checkpoint loading with "
                    "weights_only; merge aborted for safety"
                )
                self.logger.error(msg)
                return {'success': False, 'error': msg}
            source_ks = checkpoint.get('knowledge_store', {})

            if not source_ks:
                return {'success': True, 'merged': 0, 'duplicates': 0}

            # Track existing queries for deduplication
            existing_queries = set()
            if deduplicate:
                for key, value in self.model.knowledge_store.items():
                    if isinstance(value, dict):
                        query = value.get('prompt', value.get('query', str(key)))
                    else:
                        query = str(key)
                    existing_queries.add(query.lower().strip().replace(' ', ''))

            merged = 0
            duplicates = 0

            for key, value in source_ks.items():
                # Get query for dedup check
                if isinstance(value, dict):
                    query = value.get('prompt', value.get('query', str(key)))
                else:
                    query = str(key)

                query_norm = query.lower().strip().replace(' ', '')

                if deduplicate and query_norm in existing_queries:
                    duplicates += 1
                    continue

                # Add to current knowledge store
                self.model.add_to_knowledge_store(key, value)
                existing_queries.add(query_norm)
                merged += 1

            # Auto-save after merge
            self.save()

            self.logger.info(f"Merged {merged} entries from {checkpoint_path} ({duplicates} duplicates skipped)")

            return {
                'success': True,
                'merged': merged,
                'duplicates': duplicates,
                'total_knowledge': len(self.model.knowledge_store)
            }

        except Exception as e:
            self.logger.error(f"Failed to merge checkpoint: {e}")
            return {'success': False, 'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        model_stats = self.model.get_stats()

        return {
            **model_stats,
            'total_queries': self.total_queries,
            'successful_generations': self.successful_generations,
            'success_rate': self.successful_generations / max(self.total_queries, 1),
            'checkpoint_dir': self.checkpoint_dir,
            'torch_available': TORCH_AVAILABLE,
            'device': self.device
        }

    def add_knowledge(self, fact: str, category: str = "general"):
        """
        Add knowledge to LLM's knowledge base

        Args:
            fact: Mathematical fact or statement
            category: Knowledge category (theorem, definition, technique, etc.)
        """
        key = f"{category}_{len(self.model.knowledge_store)}"
        self.model.add_to_knowledge_store(key, {
            'fact': fact,
            'category': category
        })
        self.logger.debug(f"Added knowledge: {category} - {fact[:50]}...")

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get summary of stored knowledge

        Returns:
            Dictionary with knowledge statistics
        """
        return {
            'mode': 'neural' if TORCH_AVAILABLE else 'symbolic_fallback',
            'knowledge_entries': len(self.model.knowledge_store),
            **self.get_stats()
        }

    def solve_mathematical_problem(self, problem: str, problem_type: str = "general") -> Dict[str, Any]:
        """
        High-level interface for mathematical problem solving

        Args:
            problem: The mathematical problem statement
            problem_type: Type hint (calculus, algebra, proof, etc.)

        Returns:
            Dictionary with solution and metadata
        """
        task = LLMTask(
            prompt=problem,
            task_type=problem_type,
            max_tokens=512,
            temperature=0.3,  # Lower temperature for math
            metadata={'problem_type': problem_type}
        )

        response = self.handle_task(task)

        return {
            'problem': problem,
            'solution': response,
            'problem_type': problem_type,
            'stats': self.get_stats()
        }

    def clear_knowledge_store(self, reset_model: bool = False):
        """
        Clear all learned knowledge from the knowledge store.

        This is used to reset the system before a fresh learning run.

        Args:
            reset_model: If True, also reinitialize the neural model weights.
                         If False, only clears the knowledge store.
        """
        # Clear the knowledge store
        self.model.knowledge_store.clear()
        self.logger.info("Knowledge store cleared")

        # Reset statistics
        self.total_queries = 0
        self.successful_generations = 0

        if reset_model and TORCH_AVAILABLE:
            # Reinitialize model weights
            self.model = SymboLLMCore(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                num_heads=4,
                num_layers=3,
                ff_dim=512,
                max_seq_len=512,
                dropout=0.1,
                device=self.device
            )
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)
            self.logger.info("Model weights reinitialized")

        # Delete checkpoint files
        checkpoint_path = os.path.join(self.checkpoint_dir, "symbo_llm_latest.pt")
        tokenizer_path = os.path.join(self.checkpoint_dir, "tokenizer.json")

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            self.logger.info(f"Deleted checkpoint: {checkpoint_path}")

        if os.path.exists(tokenizer_path):
            os.remove(tokenizer_path)
            self.logger.info(f"Deleted tokenizer: {tokenizer_path}")

    def learn_batch(
        self,
        problems: List[Dict[str, str]],
        save_every: int = 100,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Learn from a batch of problems efficiently.

        This is optimized for bulk learning during the training phase.
        Only stores in knowledge base (no gradient updates for speed).

        Args:
            problems: List of dicts with keys:
                - 'problem_id': Unique identifier
                - 'problem_text': The problem statement
                - 'answer': The verified correct answer
                - 'domain': Problem domain (optional)
            save_every: Save checkpoint every N problems
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dict with learning statistics
        """
        if not problems:
            return {'learned': 0, 'errors': 0}

        learned = 0
        errors = 0
        total = len(problems)

        for i, problem in enumerate(problems):
            try:
                problem_id = problem.get('problem_id', f'batch_{i}')
                problem_text = problem.get('problem_text', '')
                answer = problem.get('answer', '')
                domain = problem.get('domain', 'general')

                if not problem_text or not answer:
                    errors += 1
                    continue

                # Store in knowledge base with problem_text as lookup key
                # Use normalized problem text as key for exact matching
                normalized_key = self._normalize_for_lookup(problem_text)

                self.model.add_to_knowledge_store(normalized_key, {
                    'prompt': problem_text,
                    'response': answer,
                    'category': domain,
                    'problem_id': problem_id,
                    'timestamp': datetime.now().isoformat()
                })

                learned += 1

                # Progress callback
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, total)

                # Periodic save
                if save_every > 0 and (i + 1) % save_every == 0:
                    self.save()
                    self.logger.info(f"Checkpoint saved at {i + 1}/{total} problems")

            except Exception as e:
                self.logger.error(f"Error learning problem {i}: {e}")
                errors += 1

        # Final save
        self.save()

        result = {
            'learned': learned,
            'errors': errors,
            'total_in_store': len(self.model.knowledge_store),
            'save_path': self.checkpoint_dir
        }

        self.logger.info(f"Batch learning complete: {learned} learned, {errors} errors")
        return result

    def learn_batch_with_gatekeeper(
        self,
        problems: List[Dict[str, str]],
        gatekeeper=None,
        review_queue=None,
        save_every: int = 100,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Learn from a batch of problems with gatekeeper verification.

        Applies verification before learning:
        - ACCEPT (confidence >= 0.95): Learn immediately
        - REJECT (confidence < 0.5 or validation failed): Skip
        - REVIEW (0.5-0.95): Queue for manual review
        - DUPLICATE: Skip (already learned)

        Args:
            problems: List of dicts with keys:
                - 'problem_id': Unique identifier
                - 'problem_text': The problem statement
                - 'answer': The verified correct answer
                - 'domain': Problem domain (optional)
            gatekeeper: GatekeeperSupervisor instance (or None to skip verification)
            review_queue: ReviewQueue instance for uncertain cases
            save_every: Save checkpoint every N problems
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dict with learning statistics including gatekeeper results
        """
        if not problems:
            return {
                'learned': 0, 'errors': 0, 'rejected': 0,
                'queued': 0, 'duplicates': 0
            }

        # If no gatekeeper, use legacy method
        if gatekeeper is None:
            legacy_result = self.learn_batch(problems, save_every, progress_callback)
            return {
                **legacy_result,
                'rejected': 0,
                'queued': 0,
                'duplicates': 0
            }

        # Update gatekeeper's knowledge store reference
        gatekeeper.update_knowledge_store(self.model.knowledge_store)

        learned = 0
        errors = 0
        rejected = 0
        queued = 0
        duplicates = 0
        total = len(problems)

        for i, problem in enumerate(problems):
            try:
                problem_id = problem.get('problem_id', f'batch_{i}')
                problem_text = problem.get('problem_text', '')
                answer = problem.get('answer', '')
                domain = problem.get('domain', 'general')

                if not problem_text or not answer:
                    errors += 1
                    continue

                # Verify through gatekeeper
                decision = gatekeeper.verify(
                    problem_text=problem_text,
                    answer=answer,
                    domain=domain
                )

                # Handle decision
                if decision.status.value == 'duplicate':
                    duplicates += 1
                    continue

                elif decision.status.value == 'accept':
                    # Learn the problem
                    normalized_key = self._normalize_for_lookup(problem_text)
                    self.model.add_to_knowledge_store(normalized_key, {
                        'prompt': problem_text,
                        'response': answer,
                        'category': domain,
                        'problem_id': problem_id,
                        'timestamp': datetime.now().isoformat(),
                        'gatekeeper_confidence': decision.confidence
                    })
                    learned += 1

                elif decision.status.value == 'reject':
                    rejected += 1
                    self.logger.debug(
                        f"Rejected problem {problem_id}: {decision.reason}"
                    )

                else:  # 'review'
                    queued += 1
                    if review_queue:
                        review_queue.add(
                            problem_id=problem_id,
                            problem_text=problem_text,
                            proposed_answer=answer,
                            domain=domain,
                            confidence=decision.confidence,
                            reason=decision.reason,
                            validation_details=decision.to_dict()
                        )

                # Progress callback
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, total)

                # Periodic save
                if save_every > 0 and (i + 1) % save_every == 0:
                    self.save()
                    self.logger.info(
                        f"Checkpoint saved at {i + 1}/{total} "
                        f"(learned: {learned}, rejected: {rejected}, queued: {queued})"
                    )

            except Exception as e:
                self.logger.error(f"Error processing problem {i}: {e}")
                errors += 1

        # Final save
        self.save()

        result = {
            'learned': learned,
            'errors': errors,
            'rejected': rejected,
            'queued': queued,
            'duplicates': duplicates,
            'total_in_store': len(self.model.knowledge_store),
            'save_path': self.checkpoint_dir,
            'gatekeeper_stats': gatekeeper.get_stats() if gatekeeper else None
        }

        self.logger.info(
            f"Batch learning with gatekeeper complete: "
            f"{learned} learned, {rejected} rejected, "
            f"{queued} queued for review, {duplicates} duplicates, {errors} errors"
        )
        return result

    def learn_with_enhancement(
        self,
        problems: List[Dict[str, str]],
        enhancement_supervisor=None,
        save_every: int = 100,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Learn from problems with Learning Enhancement Team optimization.

        Uses the full Learning Enhancement Team for:
        - Complexity-weighted learning (harder problems = stronger signals)
        - Negative example detection (skip known bad patterns)
        - Curriculum learning (progressive difficulty)
        - Knowledge graph integration (relationship-aware storage)

        Args:
            problems: List of dicts with keys:
                - 'problem_id': Unique identifier
                - 'problem_text': The problem statement
                - 'answer': The verified correct answer
                - 'domain': Problem domain (optional)
            enhancement_supervisor: LearningEnhancementSupervisor instance
            save_every: Save checkpoint every N problems
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dict with learning statistics including enhancement metrics
        """
        if not problems:
            return {'learned': 0, 'errors': 0, 'skipped_bad': 0, 'graph_nodes': 0}

        # If no enhancement supervisor, use legacy method
        if enhancement_supervisor is None:
            legacy_result = self.learn_batch(problems, save_every, progress_callback)
            return {
                **legacy_result,
                'skipped_bad': 0,
                'graph_nodes': 0,
                'avg_complexity': 0.0
            }

        learned = 0
        errors = 0
        skipped_bad = 0
        graph_nodes = 0
        total_complexity = 0.0
        total = len(problems)

        for i, problem in enumerate(problems):
            try:
                problem_id = problem.get('problem_id', f'batch_{i}')
                problem_text = problem.get('problem_text', '')
                answer = problem.get('answer', '')
                domain = problem.get('domain', 'general')

                if not problem_text or not answer:
                    errors += 1
                    continue

                # Run full optimization through enhancement supervisor
                optimization_result = enhancement_supervisor.optimize_learning({
                    'problem': problem_text,
                    'answer': answer,
                    'domain': domain
                })

                results = optimization_result.get('results', {})

                # Check for bad patterns
                negative_check = results.get('negative_check', {})
                if negative_check.get('is_bad', False):
                    skipped_bad += 1
                    self.logger.debug(
                        f"Skipped bad pattern for {problem_id}: {negative_check.get('reason')}"
                    )
                    continue

                # Get complexity score for weighted learning
                complexity = results.get('complexity', {})
                complexity_score = complexity.get('score', 0.5)
                learning_weight = complexity.get('learning_weight', 1.0)
                total_complexity += complexity_score

                # Store in knowledge base
                normalized_key = self._normalize_for_lookup(problem_text)
                self.model.add_to_knowledge_store(normalized_key, {
                    'prompt': problem_text,
                    'response': answer,
                    'category': domain,
                    'problem_id': problem_id,
                    'timestamp': datetime.now().isoformat(),
                    'complexity_score': complexity_score,
                    'learning_weight': learning_weight
                })
                learned += 1

                # Track knowledge graph nodes
                kg_result = results.get('knowledge_graph', {})
                if kg_result.get('node_id'):
                    graph_nodes += 1

                # Perform weighted gradient update if PyTorch available
                if TORCH_AVAILABLE and hasattr(self, 'optimizer') and self.optimizer:
                    self._weighted_gradient_update(
                        problem_text, answer, learning_weight
                    )

                # Progress callback
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, total)

                # Periodic save
                if save_every > 0 and (i + 1) % save_every == 0:
                    self.save()
                    self.logger.info(
                        f"Enhanced checkpoint saved at {i + 1}/{total} "
                        f"(learned: {learned}, skipped_bad: {skipped_bad})"
                    )

            except Exception as e:
                self.logger.error(f"Error in enhanced learning for problem {i}: {e}")
                errors += 1

        # Final save
        self.save()

        avg_complexity = total_complexity / learned if learned > 0 else 0.0

        result = {
            'learned': learned,
            'errors': errors,
            'skipped_bad': skipped_bad,
            'graph_nodes': graph_nodes,
            'avg_complexity': avg_complexity,
            'total_in_store': len(self.model.knowledge_store),
            'save_path': self.checkpoint_dir,
            'enhancement_stats': enhancement_supervisor.get_stats() if enhancement_supervisor else None
        }

        self.logger.info(
            f"Enhanced batch learning complete: "
            f"{learned} learned, {skipped_bad} skipped (bad patterns), "
            f"avg complexity: {avg_complexity:.3f}"
        )
        return result

    def _weighted_gradient_update(
        self,
        problem_text: str,
        answer: str,
        learning_weight: float
    ) -> None:
        """
        Perform a weighted gradient update for enhanced learning.

        Higher complexity problems receive stronger gradient signals.

        Args:
            problem_text: The problem text
            answer: The answer
            learning_weight: Weight multiplier for gradient (1.0 + complexity)
        """
        if not TORCH_AVAILABLE:
            return

        try:
            self.model.train()

            # Create training text
            full_text = f"Q: {problem_text}\nA: {answer}"
            tokens = self.tokenizer.encode(full_text)

            # Truncate if needed
            if len(tokens) > self.model.max_seq_len:
                tokens = tokens[:self.model.max_seq_len]

            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            target_ids = input_ids.clone()

            # Compute loss and apply weight
            loss = self.model.compute_loss(input_ids, target_ids)
            weighted_loss = loss * learning_weight

            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.logger.debug(
                f"Weighted gradient update - Loss: {loss.item():.4f}, "
                f"Weight: {learning_weight:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Weighted gradient update failed: {e}")

    def _normalize_for_lookup(self, text: str) -> str:
        """
        Normalize text for knowledge store lookup.

        Creates a consistent key for storing and retrieving problems.
        """
        # Lowercase and strip
        normalized = text.lower().strip()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        # Remove common punctuation that doesn't affect meaning
        normalized = normalized.replace('?', '').replace('.', '').replace('!', '')
        return normalized

    def query_exact(self, problem_text: str) -> Optional[str]:
        """
        Query for an exact match in the knowledge store.

        Args:
            problem_text: The problem to look up.

        Returns:
            The stored answer if found, None otherwise.
        """
        normalized = self._normalize_for_lookup(problem_text)

        # Direct lookup
        if normalized in self.model.knowledge_store:
            entry = self.model.knowledge_store[normalized]
            if isinstance(entry, dict) and 'response' in entry:
                return entry['response']

        return None

    def get_knowledge_count(self) -> int:
        """Get the number of entries in the knowledge store."""
        return len(self.model.knowledge_store)

    # =========================================================================
    # LEARNING ENHANCEMENT TEAM INTEGRATION
    # =========================================================================

    @property
    def learning_supervisor(self):
        """
        Get or create the Learning Enhancement Supervisor.

        The supervisor is lazy-loaded on first access to avoid import overhead.
        Once created, it's cached for reuse.

        Returns:
            LearningEnhancementSupervisor instance
        """
        if self._learning_supervisor is None:
            try:
                from symbo_agentic_reasoners.agents.supervisors.learning_enhancement_supervisor import (
                    LearningEnhancementSupervisor
                )
                self._learning_supervisor = LearningEnhancementSupervisor()
                self._enhancement_enabled = True
                self.logger.info("Learning Enhancement Supervisor initialized")
            except ImportError as e:
                self.logger.warning(f"Could not load Learning Enhancement Team: {e}")
                self._enhancement_enabled = False
        return self._learning_supervisor

    def enable_enhancement(self, enabled: bool = True) -> None:
        """
        Enable or disable Learning Enhancement Team integration.

        When enabled, learn_from_interaction and train methods will use:
        - Complexity scoring for weighted learning
        - Negative pattern detection
        - Curriculum learning
        - Knowledge graph integration

        Args:
            enabled: Whether to enable enhancement
        """
        self._enhancement_enabled = enabled
        if enabled and self._learning_supervisor is None:
            # Trigger lazy loading
            _ = self.learning_supervisor
        self.logger.info(f"Learning enhancement {'enabled' if enabled else 'disabled'}")

    def learn_from_interaction_enhanced(
        self,
        user_input: str,
        response: str,
        category: str = "interaction",
        domain: str = None
    ) -> Dict[str, Any]:
        """
        Learn from an interaction with Learning Enhancement Team optimization.

        This method applies the full Learning Enhancement pipeline:
        1. Check for bad patterns (skip if detected)
        2. Score complexity for weighted learning
        3. Apply weighted gradient update
        4. Store in knowledge graph with relationships

        Args:
            user_input: User's mathematical query
            response: System's response/solution
            category: Category for the interaction
            domain: Problem domain (algebra, calculus, etc.)

        Returns:
            Dict with learning results including enhancement metrics
        """
        domain = domain or category

        # If enhancement not available, fall back to regular learning
        if not self._enhancement_enabled or self.learning_supervisor is None:
            self.learn_from_interaction(user_input, response, category)
            return {'learned': True, 'enhanced': False}

        try:
            # Run full optimization through enhancement supervisor
            optimization_result = self.learning_supervisor.optimize_learning({
                'problem': user_input,
                'answer': response,
                'domain': domain
            })

            results = optimization_result.get('results', {})

            # Check for bad patterns
            negative_check = results.get('negative_check', {})
            if negative_check.get('is_bad', False):
                self.logger.debug(
                    f"Skipped bad pattern: {negative_check.get('reason')}"
                )
                return {
                    'learned': False,
                    'enhanced': True,
                    'skipped_reason': negative_check.get('reason'),
                    'complexity': 0.0
                }

            # Get complexity score
            complexity = results.get('complexity', {})
            complexity_score = complexity.get('score', 0.5)
            learning_weight = complexity.get('learning_weight', 1.0)

            # Store in knowledge base
            indexed_key = f"{category}_{len(self.model.knowledge_store)}"
            entry_data = {
                'prompt': user_input,
                'response': response,
                'category': category,
                'domain': domain,
                'timestamp': datetime.now().isoformat(),
                'complexity_score': complexity_score,
                'learning_weight': learning_weight
            }
            self.model.add_to_knowledge_store(indexed_key, entry_data)

            # Also add with normalized prompt for fast lookup
            normalized_prompt = self._normalize_for_lookup(user_input)
            self.model.add_to_knowledge_store(normalized_prompt, entry_data)

            # Perform weighted gradient update
            if TORCH_AVAILABLE and self.optimizer:
                self._weighted_gradient_update(user_input, response, learning_weight)

            # Periodically save
            if len(self.model.knowledge_store) % 10 == 0:
                self.save()

            return {
                'learned': True,
                'enhanced': True,
                'complexity_score': complexity_score,
                'learning_weight': learning_weight,
                'graph_node': results.get('knowledge_graph', {}).get('node_id')
            }

        except Exception as e:
            self.logger.error(f"Enhanced learning failed: {e}")
            # Fall back to regular learning
            self.learn_from_interaction(user_input, response, category)
            return {'learned': True, 'enhanced': False, 'error': str(e)}

    def train_with_curriculum(
        self,
        dataset: List[Dict[str, Any]],
        epochs: int = 50,
        batch_size: int = 32,
        use_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Train with curriculum learning (progressive difficulty).

        Uses the CurriculumSpecialist to select problems at appropriate
        difficulty levels, gradually increasing as the model improves.

        Args:
            dataset: List of training examples with:
                - 'prompt': Problem text
                - 'response': Answer
                - 'category': Domain/category
                - 'complexity': Optional pre-computed complexity (0.0-1.0)
            epochs: Number of training epochs
            batch_size: Batch size
            use_enhancement: Whether to use full enhancement pipeline

        Returns:
            Training statistics including curriculum progression
        """
        if not dataset:
            return {'learned': 0, 'epochs': 0}

        # If enhancement not available or disabled, use regular training
        if not use_enhancement or not self._enhancement_enabled:
            self.train(dataset, epochs, batch_size)
            return {
                'learned': len(dataset),
                'epochs': epochs,
                'curriculum': False
            }

        supervisor = self.learning_supervisor
        if supervisor is None:
            self.train(dataset, epochs, batch_size)
            return {'learned': len(dataset), 'epochs': epochs, 'curriculum': False}

        try:
            # Get curriculum specialist
            curriculum = supervisor._get_specialist('curriculum')
            complexity_scorer = supervisor._get_specialist('complexity_scorer')

            # Pre-compute complexity scores if not provided
            for item in dataset:
                if 'complexity' not in item:
                    score_result = complexity_scorer.score(
                        problem=item.get('prompt', ''),
                        answer=item.get('response', ''),
                        domain=item.get('category', 'general')
                    )
                    item['complexity'] = score_result.score

            total_learned = 0
            difficulty_history = []

            for epoch in range(epochs):
                # Get curriculum-selected batch
                batch_result = curriculum.next_batch(dataset, batch_size)
                batch = batch_result.get('batch', [])
                current_difficulty = batch_result.get('current_difficulty', 0.5)
                difficulty_history.append(current_difficulty)

                if not batch:
                    continue

                epoch_successes = 0
                for item in batch:
                    # Learn with enhancement
                    result = self.learn_from_interaction_enhanced(
                        user_input=item.get('prompt', ''),
                        response=item.get('response', ''),
                        category=item.get('category', 'general'),
                        domain=item.get('category', 'general')
                    )

                    if result.get('learned', False):
                        epoch_successes += 1
                        total_learned += 1

                # Update curriculum based on success rate
                success_rate = epoch_successes / len(batch) if batch else 0
                curriculum.update(success_rate >= 0.8)

                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Curriculum epoch {epoch + 1}/{epochs}: "
                        f"difficulty={current_difficulty:.2f}, "
                        f"success_rate={success_rate:.2%}"
                    )

            # Final save
            self.save()

            return {
                'learned': total_learned,
                'epochs': epochs,
                'curriculum': True,
                'final_difficulty': curriculum.get_state().get('current_difficulty', 0.5),
                'difficulty_history': difficulty_history,
                'curriculum_stats': curriculum.get_stats()
            }

        except Exception as e:
            self.logger.error(f"Curriculum training failed: {e}")
            self.train(dataset, epochs, batch_size)
            return {
                'learned': len(dataset),
                'epochs': epochs,
                'curriculum': False,
                'error': str(e)
            }

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the Learning Enhancement Team.

        Returns:
            Dict with enhancement team statistics including:
            - supervisor_stats: Supervisor-level metrics
            - specialist_stats: Per-specialist metrics
            - enhancement_enabled: Whether enhancement is active
        """
        if not self._enhancement_enabled or self._learning_supervisor is None:
            return {
                'enhancement_enabled': False,
                'supervisor_stats': None,
                'specialist_stats': None
            }

        try:
            return {
                'enhancement_enabled': True,
                'supervisor_stats': self._learning_supervisor.get_stats(),
                'specialist_stats': self._learning_supervisor.get_specialist_stats()
            }
        except Exception as e:
            return {
                'enhancement_enabled': True,
                'error': str(e)
            }
