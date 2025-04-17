#!/bin/bash

echo "🚀 Setting up environment for Faster R-CNN with COCO..."

#  Create necessary directories

mkdir -p datasets/coco/train2017
mkdir -p datasets/coco/val2017
mkdir -p datasets/coco/annotations


# Install Python dependencies

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete for Faster R-CNN + COCO!"
