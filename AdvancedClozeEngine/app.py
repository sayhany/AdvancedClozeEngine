"""Streamlit-based web interface for AttentionFlash."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
from .engine import AdvancedClozeEngine
import json
import base64
from datetime import datetime

@st.cache_resource
def initialize_engine(domain_tokens: Optional[List[str]] = None):
    """Initialize the AdvancedClozeEngine with caching."""
    return AdvancedClozeEngine(domain_tokens=domain_tokens)

def display_masked_text(result: Dict):
    """Display the masked text with styling."""
    st.markdown("### Generated Cloze")
    if len(result['masked'].split()) < 3:
        st.warning("⚠️ Text is very short. Consider using a longer passage for better results.")
    st.markdown(f"**{result['masked']}**")

def display_hint_badge(hint_type: str, confidence: str) -> str:
    """Generate HTML for hint badges."""
    colors = {
        'domain': {'high': '#2E7D32', 'medium': '#558B2F', 'low': '#827717'},
        'cross_sentence': {'high': '#1565C0', 'medium': '#1976D2', 'low': '#42A5F5'},
        'attention': {'high': '#4A148C', 'medium': '#6A1B9A', 'low': '#8E24AA'},
        'pattern': {'high': '#BF360C', 'medium': '#D84315', 'low': '#E64A19'}
    }
    return f"""
        <span style="
            background-color: {colors.get(hint_type, {}).get(confidence, '#757575')};
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-right: 4px;
        ">
            {hint_type.replace('_', ' ').title()}
        </span>
    """

def display_answers(result: Dict):
    """Display answers with enhanced hint visualization."""
    st.markdown("### Answers")
    for answer in result['answers']:
        with st.expander(f"MASK{answer['mask_id']}: {answer['text']}"):
            if 'hint' in answer:
                st.markdown("#### Hints by Category:")
                for hint_type, hints in answer['hint'].items():
                    if hints:
                        hint_html = "".join([
                            display_hint_badge(hint_type, hint['confidence'])
                            for hint in hints
                        ])
                        st.markdown(hint_html, unsafe_allow_html=True)
                        for hint in hints:
                            st.markdown(f"- {hint['text']}")

def plot_attention_heatmap(attention_data: List[Dict], layer_filter: Optional[List[int]] = None) -> go.Figure:
    """Create an enhanced interactive attention heatmap."""
    if layer_filter:
        attention_data = [d for d in attention_data if d['layer'] in layer_filter]
    
    layers = []
    heads = []
    scores = []
    
    for head_data in attention_data:
        layers.append(head_data['layer'])
        heads.append(head_data['head'])
        scores.append(head_data['score'])
    
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=[f"L{l}H{h}" for l, h in zip(layers, heads)],
        y=['Attention'],
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate=(
            "Layer: %{x}<br>" +
            "Score: %{z:.3f}<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Layer-Head Pairs",
        yaxis_title="Attention Score",
        dragmode='zoom',
        showlegend=False
    )
    
    return fig

def display_visualization(result: Dict):
    """Enhanced attention visualization with filtering."""
    if 'visualization' in result:
        viz = result['visualization']
        st.markdown("### Attention Analysis")
        
        # Add layer filtering
        all_layers = sorted(list({
            head['layer'] 
            for heads in viz['contributing_heads'] 
            for head in heads
        }))
        
        layer_filter = st.multiselect(
            "Filter Layers",
            options=all_layers,
            default=all_layers,
            help="Select which layers to visualize"
        )
        
        if 'contributing_heads' in viz:
            tabs = st.tabs([f"MASK{i+1}" for i in range(len(viz['contributing_heads']))])
            
            for i, (tab, heads) in enumerate(zip(tabs, viz['contributing_heads'])):
                with tab:
                    if heads:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.plotly_chart(
                                plot_attention_heatmap(heads, layer_filter),
                                use_container_width=True
                            )
                        with col2:
                            st.markdown("#### Top Contributing Heads")
                            for head in sorted(heads, key=lambda x: x['score'], reverse=True)[:3]:
                                st.markdown(
                                    f"Layer {head['layer']}, Head {head['head']}: "
                                    f"**{head['score']:.2f}**"
                                )

def handle_feedback(result: Dict):
    """Enhanced feedback handling with immediate effects."""
    st.markdown("### Feedback")
    cols = st.columns(3)
    
    feedback_effects = {
        'too_easy': "🔄 Next generation will be more challenging",
        'perfect': "✅ Current difficulty level maintained",
        'too_hard': "🔄 Next generation will be easier"
    }
    
    for col, (feedback, message) in zip(cols, feedback_effects.items()):
        with col:
            if st.button(feedback.replace('_', ' ').title()):
                st.session_state.engine.update_feedback(
                    card_id=result.get('id', 'default'),
                    outcome=feedback,
                    difficulty=result['difficulty']
                )
                st.success(message)

def get_example_texts() -> Dict[str, str]:
    """Provide example texts for different domains."""
    return {
        "General": "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
        "Medical": "The patient presented with acute respiratory symptoms, indicating potential inflammation of the bronchial passages.",
        "Technical": "The distributed system implements fault tolerance through redundant nodes and consensus protocols."
    }

def download_button(data: Dict, filename: str) -> None:
    """Create a download button for exporting data."""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'data:application/json;base64,{b64}'
    
    st.download_button(
        label="Download Results",
        data=json_str,
        file_name=filename,
        mime='application/json'
    )

def show_help_sidebar():
    """Display help information in the sidebar."""
    with st.sidebar:
        st.markdown("### 📚 Help & Information")
        
        with st.expander("What is a Cloze Card?"):
            st.markdown("""
                A cloze card is a learning tool where key words or phrases are removed from text.
                AttentionFlash uses BERT's attention patterns to intelligently choose what to mask.
            """)
        
        with st.expander("Difficulty Levels"):
            st.markdown("""
                - **Easy**: Single token, high confidence
                - **Medium**: Multiple tokens, phrase detection
                - **Hard**: Complex phrases, cross-sentence relationships
            """)
        
        with st.expander("Masking Strategies"):
            st.markdown("""
                - **Atomic**: Individual token masking
                - **Cross-Sep**: Consider relationships across sentences
            """)

def main():
    """Main Streamlit application."""
    st.title("AttentionFlash")
    show_help_sidebar()
    
    # Example text selector
    examples = get_example_texts()
    use_example = st.checkbox("Use example text")
    if use_example:
        domain = st.selectbox("Select domain", list(examples.keys()))
        text = examples[domain]
    else:
        text = st.text_area("Enter your text here", height=150)
    
    # Domain customization
    with st.expander("Domain Settings"):
        domain_text = st.text_area(
            "Enter domain-specific terms (one per line)",
            help="These terms will receive extra attention in masking"
        )
        domain_tokens = [t.strip() for t in domain_text.split('\n') if t.strip()] if domain_text else None
        
        if domain_tokens and st.button("Update Domain Terms"):
            st.session_state.engine = initialize_engine(domain_tokens)
            st.success(f"Updated with {len(domain_tokens)} domain terms")
    
    # Initialize engine if needed
    if 'engine' not in st.session_state:
        st.session_state.engine = initialize_engine()
    
    # Input validation
    st.markdown("### Input Text")
    text = st.text_area("Enter your text here", height=150)
    if text and len(text.split()) < 3:
        st.warning("⚠️ Text is too short for effective cloze generation")
    
    # Enhanced settings
    col1, col2, col3 = st.columns(3)
    with col1:
        difficulty = st.selectbox("Difficulty", ['easy', 'medium', 'hard'], index=1)
    with col2:
        strategy = st.selectbox("Strategy", ['atomic', 'cross_sep'], index=1)
    with col3:
        max_masks = st.number_input("Max Masks", 1, 5, 2)
    
    # Generation
    if st.button("Generate Cloze") and text:
        with st.spinner("Analyzing text and generating cloze..."):
            try:
                result = st.session_state.engine.generate_cloze(
                    text=text,
                    difficulty=difficulty,
                    strategy=strategy
                )
                
                if 'error' in result:
                    st.error(f"⚠️ {result['error']}")
                    st.markdown("**Suggestions:**")
                    st.markdown("- Try a longer text")
                    st.markdown("- Use more complex sentences")
                    st.markdown("- Check for proper sentence structure")
                else:
                    display_masked_text(result)
                    display_answers(result)
                    display_visualization(result)
                    handle_feedback(result)
                    
                    # Add export options
                    st.markdown("### Export Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        download_button(
                            result,
                            f"cloze_card_{timestamp}.json"
                        )
                    with col2:
                        if st.button("Generate New Version"):
                            st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("Please try again with different text or settings.")

if __name__ == "__main__":
    main()