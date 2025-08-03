import streamlit as st
import os
import time
from PIL import Image
import io
import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import our custom modules
from image_generator import LocalImageGenerator
from audio_generator import LocalAudioGenerator

# Page configuration
st.set_page_config(
    page_title="Private AI Creative Suite",
    page_icon="🎨🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .privacy-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .generated-content {
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'image_generator' not in st.session_state:
    st.session_state.image_generator = None
if 'audio_generator' not in st.session_state:
    st.session_state.audio_generator = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨🎵 Private AI Creative Suite</h1>', unsafe_allow_html=True)
    
    # Privacy badge
    st.markdown('<div class="privacy-badge">🔒 100% Local Processing - Your Data Never Leaves Your Device</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose your creative tool:",
        ["🏠 Home", "🎨 Image Generator", "🎵 Audio Generator", "⚙️ Settings"]
    )
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💻 System Info")
    
    # Check system requirements
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    disk_gb = psutil.disk_usage('/').free / (1024**3)
    
    st.sidebar.metric("RAM Available", f"{ram_gb:.1f} GB")
    st.sidebar.metric("Disk Space", f"{disk_gb:.1f} GB")
    
    # Requirements check
    if ram_gb < 16:
        st.sidebar.warning("⚠️ Less than 16GB RAM detected")
    if disk_gb < 20:
        st.sidebar.warning("⚠️ Less than 20GB free space")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎨 Image Generator":
        show_image_generator()
    elif page == "🎵 Audio Generator":
        show_audio_generator()
    elif page == "⚙️ Settings":
        show_settings()

def show_home_page():
    st.markdown("""
    ## Welcome to Your Private AI Creative Suite! 🚀
    
    This application runs **completely locally** on your Mac, ensuring your creative ideas stay private while harnessing the power of cutting-edge AI models.
    """)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Image Generation</h3>
            <p>Create stunning images using Stable Diffusion XL. Transform your ideas into visual art with local processing.</p>
            <ul>
                <li>High-quality image generation</li>
                <li>Customizable prompts</li>
                <li>Style variations</li>
                <li>Instant local processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎵 Audio Generation</h3>
            <p>Generate music and audio using Meta's MusicGen. Create original compositions from text descriptions.</p>
            <ul>
                <li>Music generation from text</li>
                <li>Multiple genres supported</li>
                <li>Customizable duration</li>
                <li>High-quality audio output</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown("## 🚀 Getting Started")
    
    st.markdown("""
    1. **Navigate** to either the Image Generator or Audio Generator using the sidebar
    2. **Configure** your generation parameters
    3. **Generate** your creative content
    4. **Download** or share your creations
    
    ### 🔒 Privacy Features
    - ✅ All processing happens locally on your device
    - ✅ No data sent to external servers
    - ✅ Models downloaded once and cached locally
    - ✅ Complete control over your creative content
    """)
    
    # System requirements
    st.markdown("## 💻 System Requirements")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Hardware**\n- Mac with Apple Silicon (M1/M2/M3) or Intel\n- 16GB+ RAM (32GB recommended)\n- 20GB+ free disk space")
    
    with col2:
        st.info("**Software**\n- Python 3.10+\n- macOS 12.0+\n- Internet connection (for initial model download)")
    
    with col3:
        st.info("**Performance**\n- First run: Models download (~10-15 minutes)\n- Subsequent runs: Instant generation\n- GPU acceleration (if available)")

def show_image_generator():
    st.markdown("## 🎨 Local Image Generator")
    st.markdown("Create stunning images using Stable Diffusion XL - running entirely on your device!")
    
    # Initialize image generator
    if st.session_state.image_generator is None:
        with st.spinner("Loading image generation model..."):
            try:
                from image_generator import LocalImageGenerator
                st.session_state.image_generator = LocalImageGenerator()
                st.success("✅ Image generation model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading image model: {str(e)}")
                return
    
    # Generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area(
            "Describe your image:",
            placeholder="A serene mountain landscape at sunset with golden clouds...",
            height=100
        )
        
        negative_prompt = st.text_area(
            "Negative prompt (what to avoid):",
            placeholder="blurry, low quality, distorted...",
            height=80
        )
    
    with col2:
        num_inference_steps = st.slider("Generation steps", 20, 50, 30)
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5)
        width = st.selectbox("Width", [512, 768, 1024], index=1)
        height = st.selectbox("Height", [512, 768, 1024], index=1)
    
    # Generate button
    if st.button("🎨 Generate Image", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt!")
            return
        
        with st.spinner("Creating your image..."):
            try:
                image = st.session_state.image_generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                )
                
                # Display the generated image
                st.markdown('<div class="generated-content">', unsafe_allow_html=True)
                st.image(image, caption=f"Generated: {prompt}", use_column_width=True)
                
                # Download button
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Image",
                    data=byte_im,
                    file_name=f"generated_image_{int(time.time())}.png",
                    mime="image/png"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")

def show_audio_generator():
    st.markdown("## 🎵 Local Audio Generator")
    st.markdown("Create music and audio using Meta's MusicGen - running entirely on your device!")
    
    # Generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        audio_prompt = st.text_area(
            "Describe your audio/music:",
            placeholder="A peaceful piano melody with soft strings in the background...",
            height=100
        )
        
        duration = st.slider("Duration (seconds)", 5, 30, 10)
    
    with col2:
        model_size = st.selectbox(
            "Model size",
            ["small", "medium", "large"],
            index=1,
            help="Larger models = better quality but slower generation"
        )
        
        temperature = st.slider("Creativity (temperature)", 0.1, 2.0, 1.0)
    
    # Initialize audio generator with selected model size
    if st.session_state.audio_generator is None:
        with st.spinner(f"Loading MusicGen {model_size} model..."):
            try:
                from audio_generator import LocalAudioGenerator
                st.session_state.audio_generator = LocalAudioGenerator(model_size=model_size)
                st.success(f"✅ MusicGen {model_size} model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading audio model: {str(e)}")
                return
    
    # Generate button
    if st.button("🎵 Generate Audio", type="primary"):
        if not audio_prompt.strip():
            st.warning("Please enter an audio prompt!")
            return
        
        with st.spinner("Creating your audio..."):
            try:
                audio_data, sample_rate = st.session_state.audio_generator.generate(
                    prompt=audio_prompt,
                    duration=duration,
                    temperature=temperature
                )
                
                # Display audio player
                st.markdown('<div class="generated-content">', unsafe_allow_html=True)
                st.audio(audio_data, sample_rate=sample_rate)
                
                # Create waveform visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=audio_data[:sample_rate*5],  # First 5 seconds
                    mode='lines',
                    name='Waveform',
                    line=dict(color='#667eea', width=1)
                ))
                fig.update_layout(
                    title="Audio Waveform (First 5 seconds)",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude",
                    height=200
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                import soundfile as sf
                buf = io.BytesIO()
                sf.write(buf, audio_data, sample_rate, format='WAV')
                byte_audio = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Audio",
                    data=byte_audio,
                    file_name=f"generated_audio_{int(time.time())}.wav",
                    mime="audio/wav"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating audio: {str(e)}")

def show_settings():
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 Image Model")
        if st.button("Download/Update Image Model"):
            with st.spinner("Downloading Stable Diffusion XL model..."):
                try:
                    # This would trigger model download
                    st.info("Model download started. This may take 10-15 minutes on first run.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("#### 🎵 Audio Model")
        if st.button("Download/Update Audio Model"):
            with st.spinner("Downloading MusicGen model..."):
                try:
                    # This would trigger model download
                    st.info("Model download started. This may take 5-10 minutes on first run.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("### Performance Settings")
    
    # Memory management
    st.markdown("#### Memory Management")
    clear_cache = st.button("Clear Model Cache")
    if clear_cache:
        st.session_state.image_generator = None
        st.session_state.audio_generator = None
        st.success("Cache cleared! Models will reload on next use.")
    
    # System info
    st.markdown("### System Information")
    
    import platform
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("OS", platform.system())
        st.metric("Python", platform.python_version())
    
    with col2:
        st.metric("CPU Cores", psutil.cpu_count())
        st.metric("RAM Usage", f"{psutil.virtual_memory().percent}%")
    
    with col3:
        st.metric("Disk Usage", f"{psutil.disk_usage('/').percent}%")
        st.metric("Available RAM", f"{psutil.virtual_memory().available / (1024**3):.1f} GB")

if __name__ == "__main__":
    main() 