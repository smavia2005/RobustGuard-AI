# RobustGuard-AI

## Overview
RobustGuard-AI is an AI-powered multimodal deepfake and fake content detection system. It can predict whether a given **Image, Video, Audio, or Text** is **Real or Fake (AI-generated / manipulated)**.

In today's world, AI-generated deepfakes are spreading fast on social media, news, and WhatsApp. RobustGuard-AI helps users, journalists, and organizations verify content authenticity and fight misinformation, fraud, and manipulation.

## Problem Statement
- Deepfake images and videos look 100% real
- AI-generated voice cloning can fake any person
- AI-generated text (ChatGPT etc.) is hard to identify
- People share fake content without verification
- Existing tools only work for one type of media

## Solution
RobustGuard-AI provides **one platform for all 4 types of media** - Image, Video, Audio, and Text - with a single unified interface and robust AI models that work even on noisy, compressed, and low-quality content.

## Key Features

### 1. Image Detection
- Detects AI-generated images (GAN / Diffusion)
- Detects face-swapped and edited images
- Works on JPG, PNG, WebP

### 2. Video Detection
- Frame-by-frame deepfake analysis
- Detects face-swap, lip-sync and face reenactment
- Works on MP4, WebM, MOV

### 3. Audio Detection
- Detects AI-cloned / synthetic voice
- Detects voice conversion and TTS-generated audio
- Works on MP3, WAV, M4A

### 4. Text Detection
- Detects AI-generated text (LLM-generated)
- Detects paraphrased and manipulated text
- Works for English and Roman Urdu / multilingual text

### 5. Robustness
- Works on noisy, blurred, and compressed media
- Adversarial attack resistant
- Low-light and low-quality handling

## How It Works
1. User uploads Image / Video / Audio / Text
2. Preprocessing: Denoising, normalization, feature extraction
3. AI Model analyzes patterns, artifacts, and inconsistencies
4. System predicts: **Real** or **Fake**
5. Shows Confidence Score + Explanation

Example Output:
> Prediction: Fake | Confidence: 94.2% | Type: Deepfake Video

## Project Structure
- backend/ - FastAPI backend, APIs, model inference logic
- frontend/ - HTML, CSS, JavaScript web interface
- training/ - Model training notebooks, datasets, evaluation scripts
- README.md - Project documentation

## Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch, TensorFlow, Transformers
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Image/Video:** OpenCV, PIL
- **Audio:** Librosa, Torchaudio
- **Text:** Hugging Face Transformers, NLTK

## Models Used
- Image: CNN / EfficientNet / Vision Transformer
- Video: CNN + LSTM / 3D-CNN + Temporal Analysis
- Audio: Spectrogram-based CNN / Wav2Vec2
- Text: BERT / RoBERTa-based Classifier

## Installation & How to Run

### 1. Clone the Repo
git clone https://github.com/smavia2005/RobustGuard-AI.git
cd RobustGuard-AI

### 2. Install Dependencies
pip install -r backend/requirements.txt

### 3. Run Backend
uvicorn backend.app:app --reload

### 4. Open Frontend
Open frontend/index.html in your browser
OR visit http://localhost:8000

## Future Enhancements
- Real-time deepfake detection for live video calls
- Browser extension for instant verification
- WhatsApp / Social media link checker
- Explainable AI - highlight fake regions in image/video
- Mobile App version

## Use Cases
- Journalists - Verify viral news before publishing
- Social Media Users - Check fake viral videos
- Banks / Companies - Verify identity / KYC videos
- Teachers / Students - Check AI-generated assignments
- Law Enforcement - Digital forensics

## Disclaimer
This tool is for educational and research purposes. Always verify with multiple sources before making legal decisions.
