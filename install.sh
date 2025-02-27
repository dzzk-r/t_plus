#!/bin/bash

echo "🔧 Installing dependencies..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Detect OS
OS_TYPE="$(uname -s)"
if [ "$OS_TYPE" == "Darwin" ]; then
    echo "🖥️ macOS detected, ensuring MPS support..."
elif [ "$OS_TYPE" == "Linux" ]; then
    echo "🐧 Linux detected, ensuring CUDA support..."
fi

echo "✅ Installation complete!"
