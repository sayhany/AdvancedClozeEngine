"""Core engine for the AdvancedClozeEngine system."""

from typing import Dict, List, Tuple, Optional, Set, Union
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
        
        # Domain-specific settings
        self.domain_tokens = domain_tokens
        self.domain_boost_factor = 1.5  # Default boost factor for domain-specific terms
        
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
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], List[str], torch.Tensor]:
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
        # Ensure attention is on the correct device
        if isinstance(attention, tuple):
            attention = tuple(a.to(self.device) for a in attention)
        else:
            attention = attention.to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return attention, tokens, outputs.last_hidden_state.to(self.device)

    def compute_token_importance(
        self, attention: Union[torch.Tensor, Tuple[torch.Tensor, ...]], layer_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        if layer_weights is None:
            layer_weights = self.layer_weights

        # Handle tuple of attention tensors
        if isinstance(attention, tuple):
            # Take the last layer's attention
            attention_tensor = attention[-1]
        else:
            attention_tensor = attention

        # Average across heads
        token_importance = attention_tensor.mean(dim=1)  # Average across heads
        token_importance = token_importance.squeeze(0)   # Remove batch dimension
        token_importance = token_importance.mean(dim=0)  # Average across sequence length

        return token_importance

    def select_mask_candidates(
        self,
        attention: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        tokens: List[str],
        text: str,
        strategy: str = "cross_sep",
        difficulty: str = "medium"
    ) -> List[Dict]:
        importance = self.compute_token_importance(attention)
        metrics = {}
        scores = importance.cpu().numpy()  # Keep scores as numpy array
        
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
            # Remove unnecessary tensor conversion
            phrases = self.detect_phrases(attention, tokens, scores, metrics)  # Pass scores directly as numpy array
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
        if isinstance(attention, tuple):
            attention_tensor = attention[-1]  # Take the last layer
        else:
            attention_tensor = attention
            
        # Average across heads for visualization
        attention_weights = attention_tensor[0].mean(dim=0).detach().cpu().numpy().tolist()
        
        viz_data = {
            "tokens": tokens,
            "attention_weights": attention_weights,
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
        attention: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        tokens: List[str],
        scores: np.ndarray,
        metrics: Dict
    ) -> List[Dict]:
        phrases = []
        seq_len = len(tokens)
        
        # Convert attention to numpy array
        if isinstance(attention, tuple):
            attention_tensor = attention[-1]  # Use the last layer if it's a tuple
        else:
            attention_tensor = attention
        
        # Average across heads (dim=1) and take the first batch
        mean_attention = attention_tensor[0].mean(dim=0).detach().cpu().numpy()
        
        # Remove unnecessary conversion since scores is already numpy array
        # scores = scores.detach().cpu().numpy()
        
        # Pre-compute factors as numpy arrays
        domain_factors = np.ones(seq_len)
        if self.domain_tokens:
            for idx, token in enumerate(tokens):
                if any(dt in token.lower() for dt in self.domain_tokens):
                    domain_factors[idx] = self.domain_boost_factor
        
        cross_sent_factors = np.ones(seq_len)
        if 'cross_attention' in metrics:
            cross_sent_factors = metrics['cross_attention'].detach().cpu().numpy()
        
        for start in range(int(seq_len) - 1):  # Convert to int for indexing
            start_idx = int(start)  # Ensure integer index
            if tokens[start_idx].startswith("["):
                continue
            
            covered_spans = set()
            
            for length in range(2, min(self.phrase_config["max_phrase_len"] + 1,
                                     int(seq_len) - start_idx + 1)):  # Convert to int for range
                end_idx = start_idx + length  # Calculate end index
                span = (start_idx, end_idx)
                
                if any(s[0] <= start_idx <= s[1] or s[0] <= end_idx <= s[1] 
                      for s in covered_spans):
                    continue
                
                # Use numpy operations with integer indices
                phrase_attention = float(np.mean(mean_attention[start_idx:end_idx, start_idx:end_idx]))
                base_score = float(np.mean(scores[start_idx:end_idx]))
                domain_bonus = float(np.mean(domain_factors[start_idx:end_idx]))
                cross_bonus = 1.0 + float(np.mean(cross_sent_factors[start_idx:end_idx]))
                
                final_score = float(base_score * (1.0 + np.log1p(domain_bonus * cross_bonus)))
                
                if final_score > self.phrase_config["phrase_threshold"]:
                    covered_spans.add(span)
                    phrases.append({
                        "start": start_idx,  # Already an integer
                        "end": end_idx,      # Already an integer
                        "tokens": tokens[start_idx:end_idx],
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
        bert_tokens = self.tokenizer.tokenize(text)
        
        # Create alignment between BERT and spaCy tokens
        bert_to_spacy = []
        spacy_idx = 0
        current_spacy_token = doc[spacy_idx].text.lower()
        
        for bert_idx, bert_token in enumerate(bert_tokens):
            # Skip special tokens
            if bert_token.startswith('['):
                bert_to_spacy.append(-1)
                continue
                
            # Clean bert token
            clean_bert = bert_token.replace('##', '').lower()
            
            # If we find a match, store the spacy index
            if clean_bert in current_spacy_token:
                bert_to_spacy.append(spacy_idx)
                current_spacy_token = current_spacy_token[len(clean_bert):]
                if not current_spacy_token:  # Move to next spacy token
                    spacy_idx += 1
                    if spacy_idx < len(doc):
                        current_spacy_token = doc[spacy_idx].text.lower()
            else:
                bert_to_spacy.append(-1)  # No alignment found
        
        # Filter candidates using the alignment
        filtered = []
        for candidate in candidates:
            if candidate["type"] == "phrase":
                start_idx = bert_to_spacy[candidate["start"]] if candidate["start"] < len(bert_to_spacy) else -1
                end_idx = bert_to_spacy[candidate["end"]-1] if candidate["end"]-1 < len(bert_to_spacy) else -1
                
                if start_idx >= 0 and end_idx >= 0:
                    if any(doc[i].pos_ not in self.skip_pos for i in range(start_idx, end_idx + 1)):
                        filtered.append(candidate)
            else:
                pos_idx = bert_to_spacy[candidate["position"]] if candidate["position"] < len(bert_to_spacy) else -1
                if pos_idx >= 0 and doc[pos_idx].pos_ not in self.skip_pos:
                    filtered.append(candidate)
        
        return filtered
    def _get_head_contributions(
        self,
        attention: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        token_idx: int
    ) -> List[Dict]:
        """Analyze which heads contributed most to token selection."""
        head_scores = []
        
        # For each layer and head
        if isinstance(attention, tuple):
            attention_tensor = attention[-1]  # Use the last layer if it's a tuple
        else:
            attention_tensor = attention

        # For single layer attention tensor
        num_heads = attention_tensor.size(1)  # heads dimension is 1 for single layer
        for head in range(num_heads):
            score = float(attention_tensor[0, head, :, int(token_idx)].mean())
            if score > 0.1:  # Only track significant contributions
                head_scores.append({
                    "layer": -1,  # Using last layer
                    "head": int(head),
                    "score": float(score)
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
