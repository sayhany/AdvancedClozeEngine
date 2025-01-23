"""Core engine for the AttentionFlash system."""

from typing import Dict, List, Tuple, Optional, Set
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from bertviz import head_view
import spacy

class AdvancedClozeEngine:
    def __init__(self, model_name: str = "bert-base-uncased", domain_tokens: Optional[List[str]] = None):
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Skip trivial POS tags
        self.skip_pos = {"DET", "PUNCT", "SPACE"}
        
        # Configuration for phrase detection
        self.phrase_config = {
            "min_phrase_len": 2,
            "max_phrase_len": 4,
            "phrase_threshold": 0.7
        }
        self.layer_weights = [0.1] * 12  # Default equal weights for all layers
        self.user_feedback = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Enhanced difficulty configuration
        self.difficulty_config = {
            'easy': {
                'num_masks': 1,
                'threshold_percentile': 95,
                'allow_phrases': False,
                'allow_overlap': False
            },
            'medium': {
                'num_masks': 2,
                'threshold_percentile': 85,
                'allow_phrases': True,
                'allow_overlap': False
            },
            'hard': {
                'num_masks': 3,
                'threshold_percentile': 75,
                'allow_phrases': True,
                'allow_overlap': True
            }
        }
        
        # Feedback adjustment parameters
        self.feedback_adjustments = {
            'too_easy': {'threshold_delta': 5, 'boost_delta': 0.1},
            'too_hard': {'threshold_delta': -5, 'boost_delta': -0.1},
            'perfect': {'threshold_delta': 0, 'boost_delta': 0}
        }
    
    def analyze_attention(
        self, text: str
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Analyze attention patterns in the input text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple containing:
            - Attention tensor
            - List of tokens
            - Hidden states
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention and tokens
        attention = outputs.attentions  # Shape: [layers, batch, heads, seq_len, seq_len]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return attention, tokens, outputs.last_hidden_state

    def compute_token_importance(
        self, attention: torch.Tensor, layer_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute importance scores for each token based on attention patterns.

        Args:
            attention: Attention tensor from model
            layer_weights: Optional weights for each layer

        Returns:
            Tensor of importance scores for each token
        """
        if layer_weights is None:
            layer_weights = self.layer_weights

        # Convert layer weights to tensor
        layer_weights = torch.tensor(layer_weights, device=self.device)
        layer_weights = layer_weights.view(-1, 1, 1, 1, 1)

        # Compute weighted attention
        weighted_attention = attention * layer_weights

        # Average across layers and heads
        token_importance = weighted_attention.mean(dim=(0, 2))  # Average across layers and heads
        token_importance = token_importance.mean(dim=1)  # Average incoming attention

        return token_importance

    def select_mask_candidates(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        text: str,
        strategy: str = "cross_sep",
        difficulty: str = "medium"
    ) -> List[Dict]:
        # Fix: compute_token_importance only takes attention and optional layer_weights
        importance = self.compute_token_importance(attention)
        metrics = {}  # Initialize empty metrics dictionary
        scores = importance.cpu().numpy()  # Convert to numpy array for easier handling
        
        # Get basic candidates
        candidates = []
        
        # Add single-token candidates
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            if not token.startswith("["):
                candidates.append({
                    "token": token,
                    "position": idx,
                    "score": float(score),
                    "type": "atomic",
                    "head_contributions": self._get_head_contributions(attention, idx)
                })
        
        # Add phrase candidates for medium/hard difficulty
        if difficulty in ["medium", "hard"]:
            phrases = self.detect_phrases(attention, tokens, torch.tensor(scores, device=self.device), metrics)
            candidates.extend(phrases)
        
        # Filter by POS tags
        candidates = self.filter_by_pos(text, candidates)
        
        # Sort by score and apply difficulty settings
        candidates.sort(key=lambda x: x["score"], reverse=True)
        max_masks = self.difficulty_config[difficulty]["num_masks"]
        
        return candidates[:max_masks]

    def generate_cloze(
        self,
        text: str,
        strategy: str = "cross_sep",
        difficulty: str = "medium"
    ) -> Dict:
        attention, tokens, hidden_states = self.analyze_attention(text)
        candidates = self.select_mask_candidates(attention, tokens, text, strategy, difficulty)
        
        if not candidates:
            return {"error": "No suitable mask candidates found"}
        
        # Create masked version
        masked_tokens = tokens.copy()
        answers = []
        
        for i, candidate in enumerate(candidates):
            if candidate["type"] == "atomic":
                masked_tokens[candidate["position"]] = f"[MASK{i+1}]"
                answer_text = candidate["token"]
            else:  # phrase
                for pos in range(candidate["start"], candidate["end"]):
                    masked_tokens[pos] = f"[MASK{i+1}]"
                answer_text = " ".join(candidate["tokens"])
            
            answers.append({
                "text": answer_text,
                "position": candidate["position"] if candidate["type"] == "atomic" 
                           else candidate["start"],
                "type": candidate["type"],
                "mask_id": i + 1,
                "head_contributions": candidate.get("head_contributions", [])
            })
        
        # Enhanced visualization data
        viz_data = {
            "tokens": tokens,
            "attention_weights": attention[-1][0].mean(0).cpu().tolist(),
            "masked_items": answers,
            "contributing_heads": [a["head_contributions"] for a in answers]
        }
        
        return {
            "original": text,
            "masked": self.tokenizer.convert_tokens_to_string(masked_tokens),
            "answers": answers,
            "difficulty": difficulty,
            "strategy": strategy,
            "visualization": viz_data
        }

    def detect_phrases(
        self,
        attention: torch.Tensor,
        tokens: List[str],
        scores: torch.Tensor,
        metrics: Dict
    ) -> List[Dict]:
        phrases = []
        seq_len = len(tokens)
        # Convert tensors to numpy arrays first
        mean_attention = attention.mean(dim=(0, 2))[0].detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        
        # Pre-compute factors as numpy arrays
        domain_factors = np.ones(seq_len)
        if self.domain_tokens:
            for idx, token in enumerate(tokens):
                if token in self.domain_tokens:
                    domain_factors[idx] = self.domain_boost_factor
        
        cross_sent_factors = np.ones(seq_len)
        if 'cross_attention' in metrics:
            cross_sent_factors = metrics['cross_attention'].detach().cpu().numpy()
        
        for start in range(seq_len - 1):
            if tokens[start].startswith("["):
                continue
            
            covered_spans = set()
            
            for length in range(2, min(self.phrase_config["max_phrase_len"] + 1,
                                     seq_len - start + 1)):
                end = start + length
                span = (start, end)
                
                if any(s[0] <= start <= s[1] or s[0] <= end <= s[1] 
                      for s in covered_spans):
                    continue
                
                # Use numpy operations
                phrase_attention = float(np.mean(mean_attention[start:end, start:end]))
                base_score = float(np.mean(scores[start:end]))
                domain_bonus = float(np.mean(domain_factors[start:end]))
                cross_bonus = 1.0 + float(np.mean(cross_sent_factors[start:end]))
                
                final_score = base_score * (1.0 + np.log1p(domain_bonus * cross_bonus))
                
                if final_score > self.phrase_config["phrase_threshold"]:
                    covered_spans.add(span)
                    phrases.append({
                        "start": int(start),  # Ensure integer indices
                        "end": int(end),      # Ensure integer indices
                        "tokens": tokens[start:end],
                        "score": float(final_score),
                        "type": "phrase",
                        "domain_bonus": float(domain_bonus),
                        "cross_sent_bonus": float(cross_bonus),
                        "attention_pattern": phrase_attention,
                        "hint": self.generate_hint({"type": "phrase", "score": final_score}, metrics)
                    })
        
        return phrases
    
        def filter_by_pos(self, text: str, candidates: List[Dict]) -> List[Dict]:
            """Filter candidates based on POS tags."""
            doc = self.nlp(text)
            token_pos = [token.pos_ for token in doc]
            
            # Align BERT tokens with spaCy tokens (simplified alignment)
            filtered = []
            for candidate in candidates:
                if candidate["type"] == "phrase":
                    # For phrases, check if any token is important
                    start_idx = candidate["start"]
                    end_idx = candidate["end"]
                    if any(pos not in self.skip_pos 
                          for pos in token_pos[start_idx:end_idx]):
                        filtered.append(candidate)
                else:
                    # For single tokens
                    pos = token_pos[candidate["position"]]
                    if pos not in self.skip_pos:
                        filtered.append(candidate)
            
            return filtered
    def _get_head_contributions(
            self,
            attention: torch.Tensor,
            token_idx: int
        ) -> List[Dict]:
            """Analyze which heads contributed most to token selection."""
            head_scores = []
            
            # For each layer and head
            for layer in range(attention.size(0)):
                for head in range(attention.size(2)):
                    score = float(attention[layer, 0, head, :, token_idx].mean())
                    if score > 0.1:  # Only track significant contributions
                        head_scores.append({
                            "layer": layer,
                            "head": head,
                            "score": score
                        })
            
            # Sort by score and return top contributors
            head_scores.sort(key=lambda x: x["score"], reverse=True)
            return head_scores[:3]  # Return top 3 contributing heads
    def generate_hint(
        self,
        candidate: Dict,
        metrics: Dict
    ) -> Dict:
        """Generate structured hints with confidence levels."""
        hints = {
            "attention": [],
            "domain": [],
            "cross_sentence": [],
            "pattern": []
        }
        
        # Head contribution analysis
        if "head_contributions" in candidate:
            top_heads = candidate["head_contributions"]
            if len(top_heads) >= 2:
                primary = top_heads[0]
                secondary = top_heads[1]
                if primary["score"] > 0.3:
                    hints["attention"].append({
                        "type": "strong_head",
                        "text": f"Strong focus from L{primary['layer']}H{primary['head']}",
                        "confidence": "high"
                    })
                if secondary["score"] > 0.2:
                    hints["attention"].append({
                        "type": "supporting_head",
                        "text": f"Support from L{secondary['layer']}H{secondary['head']}",
                        "confidence": "medium"
                    })
        
        # Domain relevance with granular levels
        domain_bonus = candidate.get("domain_bonus", 1.0)
        if domain_bonus > 1.0:
            confidence = "high" if domain_bonus > 1.5 else "medium"
            hints["domain"].append({
                "type": "domain_match",
                "text": "Highly domain-specific" if confidence == "high" else "Domain-relevant",
                "confidence": confidence
            })
        
        # Cross-sentence analysis
        cross_bonus = candidate.get("cross_sent_bonus", 1.0)
        if cross_bonus > 1.0:
            level = "strong" if cross_bonus > 1.5 else "moderate"
            hints["cross_sentence"].append({
                "type": "cross_sentence",
                "text": f"{level.title()} connection across sentences",
                "confidence": "high" if level == "strong" else "medium"
            })
        
        # Attention pattern analysis
        if "attention_pattern" in candidate:
            pattern = candidate["attention_pattern"]
            if pattern > 0.8:
                hints["pattern"].append({
                    "type": "focused",
                    "text": "Very focused attention pattern",
                    "confidence": "high"
                })
            elif pattern < 0.3:
                hints["pattern"].append({
                    "type": "diffuse",
                    "text": "Broadly distributed attention",
                    "confidence": "low"
                })
        
        return hints