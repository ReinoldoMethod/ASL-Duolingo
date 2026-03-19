---
title: ASL Alphabet Teacher
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: false
---

# ASL Alphabet Teacher

Learn the ASL (American Sign Language) alphabet using your webcam! This app uses MediaPipe for hand detection and a RandomForest classifier to recognize ASL letters.

## Features
- **Free Sign**: Sign any learned letter and see real-time detection
- **Learn Mode**: Step-by-step guide through letters A, B, C
- **Practice Mode**: Quiz yourself with random letters and track your score

## How to Use
1. Allow webcam access when prompted
2. Face your palm toward the camera
3. Ensure good lighting
4. Hold your hand at arm's length from the camera

## MVP Scope
Currently supports letters **A, B, C**. More letters can be added by collecting additional training data.

## Tech Stack
- MediaPipe Hands for hand landmark detection
- scikit-learn RandomForest for letter classification
- Gradio for the web interface
