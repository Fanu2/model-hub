# 🤗 Hugging Face Model Downloader & Tester

This Streamlit app allows you to:

- Download Hugging Face models locally.
- Automatically load them for inference.
- Test **text**, **vision**, and **audio (ASR)** models interactively.

---

## Features

1. **Download Models**  
   Download any Hugging Face model to a local folder.

2. **Automatic Loading**  
   Detects if the model is:
   - Text (causal LM)
   - Generic Transformer / embedding
   - Vision (feature extractor)
   - Audio ASR (Whisper, Wav2Vec2)

3. **Test Models Immediately**
   - **Text models**: Enter prompt → Generate text  
   - **Vision models**: Upload image → Extract features  
   - **Audio ASR models**: Upload audio → Get transcription

4. **Supports 30 Popular Models + manual input**  
   Includes models like GPT-2, LLaMA 2, Qwen, Stable Diffusion, Whisper, Wav2Vec2, and more.

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
