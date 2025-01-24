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

## Using the Streamlit Interface

The AdvancedClozeEngine comes with a user-friendly web interface built with Streamlit, making it easy to generate and visualize cloze tests without writing any code.

### Quick Start with Streamlit

1. Make sure you've completed the installation steps above
2. Navigate to the project directory:
```bash
cd AdvancedClozeEngine
```

3. Activate the virtual environment:
```bash
# On macOS/Linux:
source myenv/bin/activate

# On Windows:
myenv\Scripts\activate
```

4. Launch the Streamlit app:
```bash
streamlit run AdvancedClozeEngine/app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Enter Text**: Type or paste your text in the input area. You can also use the example texts provided.

2. **Configure Settings**:
   - **Difficulty**: Choose between Easy, Medium, or Hard
   - **Strategy**: Select the masking strategy
   - **Max Masks**: Adjust the maximum number of words/phrases to mask

3. **Domain Settings** (Optional):
   - Click "Domain Settings" to expand
   - Enter domain-specific terms (one per line) to boost their masking probability

4. **Generate and Review**:
   - Click "Generate Cloze" to create the test
   - Review the masked text and answers
   - Analyze attention patterns in the visualization
   - Provide feedback to improve future generations

### Features of the Web Interface

- **Interactive Visualization**: View attention patterns and head contributions
- **Instant Feedback**: Rate the difficulty as "Too Easy", "Perfect", or "Too Hard"
- **Export Options**: Download results as JSON or generate new versions
- **Domain Customization**: Add domain-specific terms for better masking
- **Multiple Example Texts**: Try different domains (General, Medical, Technical)

### Troubleshooting

If you encounter any issues:

1. **ModuleNotFoundError**:
   - Ensure you're in the virtual environment
   - Try reinstalling requirements: `pip install -r requirements.txt`

2. **CUDA/GPU Issues**:
   - The engine automatically falls back to CPU if CUDA is unavailable
   - No additional configuration needed

3. **Memory Issues**:
   - Try shorter text segments
   - Reduce the maximum number of masks

4. **Visualization Problems**:
   - Make sure you have a modern web browser
   - Clear browser cache if visualizations don't load

### Tips for Best Results

1. **Text Length**: 
   - Optimal results with 1-3 sentences
   - Longer texts may take more processing time

2. **Difficulty Selection**:
   - Start with "Medium" difficulty
   - Use feedback to adjust difficulty level

3. **Domain Adaptation**:
   - Add relevant technical terms for specialized texts
   - Use 3-5 domain terms for best results

4. **Visualization**:
   - Use the layer filtering to focus on specific attention patterns
   - Compare different masks' attention patterns

For more advanced usage and API documentation, see the sections below.

### Deploying to Streamlit Cloud

You can easily deploy the app to Streamlit Cloud to share it with others:

1. **Fork the Repository**:
   - Fork this repository to your GitHub account
   - Make sure your fork is public

2. **Sign up for Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy the App**:
   - Click "New app"
   - Select your forked repository
   - Set the main file path to: `AdvancedClozeEngine/app.py`
   - Click "Deploy"

The app will be available at a public URL that you can share with others. Streamlit Cloud will automatically:
- Install all dependencies from requirements.txt
- Set up the Python environment
- Handle HTTPS and scaling

**Note**: The free tier of Streamlit Cloud has some limitations:
- Memory usage is limited
- Compute resources are shared
- App goes to sleep after inactivity

For production use or larger workloads, consider upgrading to a paid tier or hosting on your own infrastructure.
