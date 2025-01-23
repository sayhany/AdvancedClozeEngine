# AdvancedClozeEngine

AdvancedClozeEngine is a sophisticated cloze test generation system powered by BERT-based attention analysis. It creates intelligent gap-fill exercises by analyzing text using transformer attention patterns and advanced natural language processing techniques.

## Features

- **Intelligent Gap Selection**: Uses BERT attention patterns to identify the most meaningful words and phrases to mask
- **Multi-level Difficulty**: Supports easy, medium, and hard difficulty levels with configurable parameters
- **Phrase Detection**: Advanced phrase detection for creating more challenging and contextually relevant cloze tests
- **Attention Visualization**: Provides detailed visualization of attention patterns and head contributions
- **Domain Adaptation**: Supports domain-specific token boosting for specialized content
- **Smart Hint Generation**: Generates structured hints based on attention patterns and linguistic features
- **Interactive Web Interface**: Built-in Streamlit UI for easy interaction and visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AdvancedClozeEngine.git
cd AdvancedClozeEngine
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from AdvancedClozeEngine import AdvancedClozeEngine

# Initialize the engine
engine = AdvancedClozeEngine()

# Generate a cloze test
text = "The quick brown fox jumps over the lazy dog."
result = engine.generate_cloze(text, difficulty="medium")

# Access the results
print("Masked text:", result["masked"])
print("Answers:", result["answers"])
```

### Difficulty Levels

The engine supports three difficulty levels:

- **Easy**: Single word masking with high confidence threshold
- **Medium**: Multiple words and basic phrase masking
- **Hard**: Complex phrase masking with overlapping allowed

```python
# Generate cloze tests with different difficulty levels
easy_cloze = engine.generate_cloze(text, difficulty="easy")
medium_cloze = engine.generate_cloze(text, difficulty="medium")
hard_cloze = engine.generate_cloze(text, difficulty="hard")
```

### Advanced Features

#### Domain Adaptation

```python
# Initialize with domain-specific tokens
domain_tokens = ["python", "programming", "code"]
engine = AdvancedClozeEngine(domain_tokens=domain_tokens)
```

#### Attention Analysis

```python
# Get detailed attention analysis
attention, tokens, hidden_states = engine.analyze_attention(text)

# Get head contributions for specific tokens
head_info = engine._get_head_contributions(attention, token_idx=1)
```

## Configuration

The engine provides various configuration options:

```python
# Customize phrase detection parameters
engine.phrase_config = {
    "min_phrase_len": 2,
    "max_phrase_len": 4,
    "phrase_threshold": 0.7
}

# Adjust difficulty settings
engine.difficulty_config["medium"] = {
    "num_masks": 2,
    "threshold_percentile": 85,
    "allow_phrases": True,
    "allow_overlap": False
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on Hugging Face's Transformers library
- Uses BERT-based attention analysis
- Incorporates spaCy for linguistic analysis

## Running the Web Interface

Start the Streamlit interface:
```bash
cd AdvancedClozeEngine
streamlit run AdvancedClozeEngine/app.py
```