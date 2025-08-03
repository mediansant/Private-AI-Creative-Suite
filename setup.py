#!/usr/bin/env python3
"""
Setup script for Private AI Creative Suite
Automates the installation and setup process
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_system_requirements():
    """Check system requirements."""
    print("\n🔍 Checking system requirements...")
    
    # Check OS
    if platform.system() != "Darwin":
        print("⚠️  This application is optimized for macOS. Other systems may work but are not tested.")
    
    # Check available memory and disk space
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"📊 Available RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 16:
            print("⚠️  Warning: Less than 16GB RAM detected. Performance may be limited.")
        else:
            print("✅ RAM requirement met")
        
        # Check disk space
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        print(f"💾 Available disk space: {disk_gb:.1f} GB")
        
        if disk_gb < 20:
            print("⚠️  Warning: Less than 20GB free space. Model downloads may fail.")
        else:
            print("✅ Disk space requirement met")
            
    except ImportError:
        print("⚠️  Could not check system resources (psutil not available)")
        print("   This is normal - psutil will be installed with other dependencies")

def create_virtual_environment():
    """Create a virtual environment."""
    print("\n🐍 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the appropriate pip command for the virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"

def get_python_command():
    """Get the appropriate python command for the virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\python"
    else:
        return "venv/bin/python"

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    try:
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print("✅ Pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not upgrade pip: {e}")
    
    # Install requirements
    try:
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "generated_images",
        "generated_audio",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def create_launch_script():
    """Create a launch script for easy startup."""
    print("\n🚀 Creating launch script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting Private AI Creative Suite...
call venv\\Scripts\\activate
streamlit run app.py
pause
"""
        script_path = "launch.bat"
    else:
        script_content = """#!/bin/bash
echo "Starting Private AI Creative Suite..."
source venv/bin/activate
streamlit run app.py
"""
        script_path = "launch.sh"
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    print(f"✅ Launch script created: {script_path}")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    python_cmd = get_python_command()
    
    # Test imports
    test_script = """
import sys
print("Testing imports...")

try:
    import streamlit
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from diffusers import StableDiffusionXLPipeline
    print("✅ Diffusers imported successfully")
except ImportError as e:
    print(f"❌ Diffusers import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")
    sys.exit(1)

print("✅ All imports successful!")
"""
    
    try:
        result = subprocess.run([python_cmd, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n🚀 To start the application:")
    
    if platform.system() == "Windows":
        print("   Double-click 'launch.bat' or run:")
        print("   launch.bat")
    else:
        print("   Run the launch script:")
        print("   ./launch.sh")
    
    print("\n   Or manually:")
    print("   source venv/bin/activate")
    print("   streamlit run app.py")
    
    print("\n📖 First time setup:")
    print("   1. The application will open in your browser")
    print("   2. Models will download automatically (~10-15 minutes)")
    print("   3. Once loaded, you can start generating content!")
    
    print("\n📚 For more information:")
    print("   - Read the README.md file")
    print("   - Check the troubleshooting section if you encounter issues")
    
    print("\n🔒 Privacy reminder:")
    print("   - All processing happens locally on your device")
    print("   - No data is sent to external servers")
    print("   - Models are cached locally after first download")
    
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🎨🎵 Private AI Creative Suite - Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create launch script
    create_launch_script()
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Installation test failed. Please check the errors above.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 