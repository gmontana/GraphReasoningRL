#!/bin/bash
# Setup script for Codex environment
# Installs dependencies and downloads dataset before network access is disabled

set -e

# Install Python dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Install package in editable mode
pip install -e .

# Download dataset repository and copy NELL-995
if [ ! -d "KB-Reasoning-Data" ]; then
    git clone https://github.com/wenhuchen/KB-Reasoning-Data.git
fi
cp -r KB-Reasoning-Data/NELL-995 ./

