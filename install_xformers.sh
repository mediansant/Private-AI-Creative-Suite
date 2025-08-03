#!/bin/bash
echo "🔧 Installing xformers for memory optimization..."

# Activate virtual environment
source venv/bin/activate

# Install xformers
echo "📦 Installing xformers..."
pip install xformers

echo "✅ Xformers installed successfully!"
echo "🔄 Restart your application to use the memory optimization:"
echo "   ./launch.sh" 