# 🚀 Quick Start Guide

Get your Private AI Creative Suite up and running in minutes!

## ⚡ Super Quick Setup (3 steps)

### 1. Run the Setup Script
```bash
python3 setup.py
```

This will:
- ✅ Check your system requirements
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Create necessary directories
- ✅ Test the installation

### 2. Launch the Application
```bash
./launch.sh
```

Or manually:
```bash
source venv/bin/activate
streamlit run app.py
```

### 3. Start Creating!
- 🌐 The app opens in your browser at `http://localhost:8501`
- 📥 Models download automatically on first use (~10-15 minutes)
- 🎨 Start generating images and audio!

## 🎯 What You Can Do Right Now

### Generate Images
1. Click **"🎨 Image Generator"** in the sidebar
2. Type: `"A beautiful sunset over mountains, digital art style"`
3. Click **"🎨 Generate Image"**
4. Download your creation!

### Generate Audio
1. Click **"🎵 Audio Generator"** in the sidebar
2. Type: `"A peaceful piano melody with soft strings"`
3. Click **"🎵 Generate Audio"**
4. Listen and download!

## 🔧 If You Encounter Issues

### Common Solutions

**"Module not found" errors:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Out of memory errors:**
- Use smaller model sizes
- Close other applications
- Reduce generation parameters

**Slow generation:**
- Use smaller models
- Reduce generation steps
- Ensure GPU acceleration is enabled

### System Requirements Check
- **RAM**: At least 16GB (32GB recommended)
- **Storage**: 20GB+ free space
- **Python**: 3.10 or higher
- **OS**: macOS 12.0+ (optimized for Apple Silicon)

## 🎨 Pro Tips

### Better Image Prompts
- Be specific: `"A serene mountain landscape at sunset with golden clouds"`
- Add style: `"digital art", "photorealistic", "watercolor"`
- Avoid issues: `"blurry, low quality, distorted"`

### Better Audio Prompts
- Include genre: `"jazz piano", "electronic dance music"`
- Describe instruments: `"acoustic guitar", "synthesizer"`
- Mention mood: `"peaceful", "energetic", "melancholic"`

## 🔒 Privacy Features

- ✅ **100% Local**: No data leaves your device
- ✅ **Model Caching**: Downloaded once, cached locally
- ✅ **No Tracking**: No analytics or user data collection
- ✅ **Complete Control**: Your content stays private

## 📞 Need Help?

1. Check the full [README.md](README.md) for detailed documentation
2. Review the troubleshooting section
3. Ensure your system meets the requirements
4. Try clearing the model cache in Settings

---

**Ready to create? Let's go! 🎨🎵✨** 