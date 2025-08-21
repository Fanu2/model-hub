import streamlit as st
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoFeatureExtractor, pipeline
)
import torch
from PIL import Image
import os
from pathlib import Path
from io import BytesIO
import soundfile as sf

st.set_page_config(page_title="HF Model Downloader & Tester", page_icon="🤗", layout="centered")
st.title("🤗 Hugging Face Model Downloader & Tester")
st.write("Download models locally, load them, and test text, vision, or audio models immediately.")

# --------------------------
# Popular Models
# --------------------------
popular_models = [
    "gpt2","bert-base-uncased","distilbert-base-uncased","facebook/bart-large",
    "google/flan-t5-base","google/flan-t5-xxl","bigscience/bloom-560m",
    "bigscience/bloomz-1b7","tiiuae/falcon-7b-instruct","meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2","Qwen/Qwen-7B-Chat",
    "NousResearch/Llama-2-13b-chat-hf","OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "togethercomputer/RedPajama-INCITE-7B-Chat","sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base","thenlper/gte-large","Alibaba-NLP/gte-Qwen2-7B-instruct",
    "jinaai/jina-embeddings-v2-base-en","openai/clip-vit-base-patch16",
    "runwayml/stable-diffusion-v1-5","stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0","CompVis/ldm-text2im-large-256",
    "openai/whisper-base","openai/whisper-large-v2","facebook/wav2vec2-base-960h",
    "speechbrain/asr-transformer-transformerlm-librispeech","espnet/kan-bayashi_ljspeech_vits"
]

choice = st.radio("Choose input method:", ["Pick from dropdown", "Enter manually"])
if choice == "Pick from dropdown":
    model_id = st.selectbox("Select a model (searchable):", popular_models, index=0)
else:
    model_id = st.text_input("Enter Hugging Face model ID:")

save_dir = st.text_input("Enter local folder path to save the model:", "./models")

# --------------------------
# Download, Load, Test
# --------------------------
if st.button("Download, Load, and Test"):
    if not model_id.strip():
        st.error("⚠️ Please select or enter a valid model ID.")
    else:
        try:
            st.write(f"📥 Downloading `{model_id}` ...")
            local_model_path = snapshot_download(repo_id=model_id, cache_dir=save_dir)
            st.success(f"✅ Download completed: {local_model_path}")

            # --------------------------
            # Auto-detect model type
            # --------------------------
            model_type = "unknown"
            tokenizer = None
            model = None
            feature_extractor = None
            audio_pipeline = None

            # Text models
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = AutoModelForCausalLM.from_pretrained(local_model_path)
                model_type = "text"
                st.success("✅ Loaded as language model.")
            except Exception:
                try:
                    model = AutoModel.from_pretrained(local_model_path)
                    model_type = "embedding"
                    st.success("✅ Loaded as generic transformer.")
                except Exception:
                    # Vision / Audio
                    try:
                        feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
                        # Detect if audio
                        if "audio" in model_id.lower() or "whisper" in model_id.lower() or "wav2vec" in model_id.lower():
                            audio_pipeline = pipeline(task="automatic-speech-recognition", model=local_model_path)
                            model_type = "audio_asr"
                            st.success("✅ Loaded as audio ASR model.")
                        else:
                            model_type = "vision"
                            st.success("✅ Loaded as feature extractor (vision).")
                    except Exception:
                        st.error("❌ Could not auto-load model.")

            # --------------------------
            # Testing section
            # --------------------------
            st.markdown("---")
            st.header("Test the model")

            if model_type == "text":
                prompt = st.text_area("Enter prompt for text generation:", "Hello, Hugging Face!")
                max_tokens = st.slider("Max tokens to generate:", 10, 200, 50)
                if st.button("Run Text Model"):
                    st.write("Generating text...")
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.text_area("Generated Text:", generated_text, height=200)

            elif model_type == "vision":
                uploaded_file = st.file_uploader("Upload an image to test the vision model", type=["png","jpg","jpeg"])
                if uploaded_file:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.write("Extracting features...")
                    features = feature_extractor(images=image, return_tensors="pt")
                    st.write("Feature shape:", features.pixel_values.shape)
                    st.success("✅ Features extracted.")

            elif model_type == "audio_asr":
                uploaded_audio = st.file_uploader("Upload audio to transcribe", type=["wav", "mp3", "flac"])
                if uploaded_audio:
                    st.audio(uploaded_audio, format="audio/wav")
                    st.write("Transcribing audio...")
                    transcription = audio_pipeline(uploaded_audio)
                    st.text_area("Transcription:", transcription['text'], height=150)

            else:
                st.info("Model loaded but interactive testing is not implemented for this model type.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
