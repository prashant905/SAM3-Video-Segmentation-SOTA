# SAM3 Video Object Segmentation & SOTA AI Replacement

This application provides a state-of-the-art (SOTA) pipeline for segmenting objects in videos and replacing them with AI-generated content (e.g., from Gemini or other models). It addresses common issues like flickering, halos, and "flat" look by implementing advanced computer vision techniques.

## 🌟 Key Features

### 1. Robust Video Segmentation
- Uses **SAM 3 (Segment Anything Model 3)** for high-quality object tracking.
- Supports Point and Box prompting to select objects.
- Propagates masks across the entire video automatically.

### 2. SOTA AI Object Replacement
- **Geometry-Guided Propagation:** Uses **Optical Flow** and **Homography** to track the object's 3D perspective (tilt, rotation) instead of just 2D scaling. This makes the inserted object look like it's actually on the surface (e.g., a poster on a wall).
- **Robust Fallback:** If tracking fails, it falls back to a stable **Axis-Aligned Bounding Box (AABB)** method to ensure the object never disappears or flickers.
- **Color Harmonization:** Automatically adjusts the brightness and contrast of the inserted object to match the lighting of the video scene, preventing the "glowing sticker" effect.
- **Halo Removal:** Uses mask erosion and alpha blending to eliminate white radiating edges around the inserted object.

### 3. Audio Preservation
- The final rendered video retains the **original audio** from the source video using FFmpeg merging.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for SAM3)
- FFmpeg (for audio merging)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/prashant905/SAM3-Video-Segmentation-SOTA.git
   cd SAM3-Video-Segmentation-SOTA
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the App

1. Start the Gradio interface:
   ```bash
   python app.py
   ```

2. Open your browser and go to:
   ```
   http://localhost:7860
   ```

## 📖 Usage Workflow

1. **Upload Video:** Upload the video you want to edit.
2. **Segment Object:** Click on the object you want to replace (Point Prompt) or draw a box around it.
3. **Propagate Masks:** Click **"Propagate across video"** (Orange Button). Wait for SAM3 to track the object through all frames.
4. **Generate/Upload Replacement:** Use the "Gemini Image Generation" tab to create a new object (e.g., "Coca Cola Ad") or upload your own image.
5. **Propagate AI Edit:** Click **"🚀 Propagate AI Edit Across Video"**. This applies the SOTA replacement pipeline (Homography + Harmonization).
6. **Render:** Click **"Render Video"** to download the final result with audio.

## ☁️ Deployment (Server)

To keep the app running on a server without it stopping when you disconnect:

1. Install `tmux`:
   ```bash
   sudo apt-get install tmux
   ```

2. Start a session:
   ```bash
   tmux new -s video_app
   ```

3. Run the app inside the session:
   ```bash
   source venv/bin/activate
   python app.py
   ```

4. Detach (Ctrl+B, then D). The app will keep running in the background.

## 📄 License
MIT License
