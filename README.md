# AttentionFlash

AttentionFlash is a next-generation cloze flashcard system that leverages BERT's attention mechanisms to intelligently generate study materials. The system analyzes text using transformer attention patterns to identify key concepts and create effective learning exercises.

## Features

- **Intelligent Masking**: Uses BERT attention patterns to identify important concepts
- **Multi-difficulty Levels**: Supports easy, medium, and hard difficulty settings
- **Phrase Detection**: Identifies meaningful phrases beyond single tokens
- **Attention Visualization**: Provides insights into model decision-making
- **Adaptive Difficulty**: Adjusts based on user feedback
- **Domain Customization**: Supports domain-specific vocabulary emphasis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from attention_flash import create_engine

# Initialize the engine
engine = create_engine()

# Generate a cloze exercise
result = engine.generate_cloze(
    text="The transformer architecture revolutionized natural language processing.",
    difficulty="medium"
)

print(f"Question: {result['masked']}")
print(f"Answer: {result['answers'][0]['text']}")
```

## API Reference

### AdvancedClozeEngine

The core class that handles cloze generation and attention analysis.

#### Constructor

```python
AdvancedClozeEngine(model_name: str = "bert-base-uncased", domain_tokens: Optional[List[str]] = None)
```

- `model_name`: The BERT model to use (default: "bert-base-uncased")
- `domain_tokens`: Optional list of domain-specific tokens to emphasize

#### Methods

##### generate_cloze

```python
generate_cloze(text: str, strategy: str = "cross_sep", difficulty: str = "medium") -> Dict
```

Generates a cloze exercise from the input text.

- `text`: Input text to create exercise from
- `strategy`: Attention analysis strategy (default: "cross_sep")
- `difficulty`: Exercise difficulty level ("easy", "medium", "hard")

Returns a dictionary containing:
- `original`: Original input text
- `masked`: Text with masked tokens
- `answers`: List of correct answers and their positions
- `visualization`: Attention visualization data

##### analyze_attention

```python
analyze_attention(text: str) -> Tuple[torch.Tensor, List[str], torch.Tensor]
```

Analyzes attention patterns in the input text.

- `text`: Input text to analyze

Returns:
- Attention tensor
- List of tokens
- Hidden states

## Difficulty Levels

### Easy
- Single token masking
- High confidence selections (95th percentile)
- No phrase masking
- No overlapping masks

### Medium
- Two token masking
- Moderate confidence selections (85th percentile)
- Allows phrase masking
- No overlapping masks

### Hard
- Three token masking
- Lower confidence threshold (75th percentile)
- Allows phrase masking
- Allows overlapping masks

## Technical Details

### Attention Analysis

The system analyzes attention patterns across all layers and heads of the BERT model to identify important tokens and phrases. Key components include:

1. **Token Importance Scoring**: Combines attention weights across layers and heads
2. **Phrase Detection**: Identifies meaningful multi-token sequences
3. **Head Contribution Analysis**: Tracks which attention heads contribute most to token selection

### Hint Generation

The system generates structured hints based on:

- Attention patterns
- Domain relevance
- Cross-sentence relationships
- Head contribution analysis

Hints include confidence levels and specific attention head information.

## License

MIT License

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- BertViz 1.4+
- Additional dependencies in requirements.txt