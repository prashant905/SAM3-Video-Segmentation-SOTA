# 🎥 SAM3 Video Segmentation & SOTA Object Replacement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SAM 3](https://img.shields.io/badge/Model-SAM%203-green.svg)](https://github.com/facebookresearch/segment-anything)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art (SOTA) video editing application that leverages **Segment Anything Model 3 (SAM 3)** for robust object tracking and **Generative AI** for seamless object replacement. This tool solves common video editing challenges like flickering, halos, and flat perspectives using advanced computer vision techniques.

---

## 🌟 Key Features

### 🧠 Advanced Segmentation
- **SAM 3 Integration:** Utilizes the latest Segment Anything Model for pixel-perfect object tracking across video frames.
- **Multi-Modal Prompting:** Select objects using **Points** (click) or **Bounding Boxes** (drag).
- **Automatic Propagation:** Track objects through occlusions and motion with high temporal consistency.

### 🎨 SOTA Object Replacement Pipeline
This isn't just a simple "copy-paste". We implement a sophisticated pipeline to ensure realism:

1.  **Geometry-Guided Warping (Homography):**
    *   Instead of simple 2D resizing, we track feature points on the object surface.
    *   We compute a **3D Homography Matrix** to tilt, rotate, and slant the inserted object (e.g., a poster) to match the perspective of the original surface.
    *   *Result:* The new object looks "painted on" the surface, not floating above it.

2.  **Color Harmonization:**
    *   The app analyzes the lighting and color statistics of the target scene.
    *   It dynamically adjusts the **Luminance (L-channel)** and contrast of the inserted object to match the environment.
    *   *Result:* No "glowing sticker" effect; the object blends naturally into shadows and highlights.

3.  **Smart Blending & Halo Removal:**
    *   **Mask Erosion:** Shrinks the mask slightly to remove the original object's fringe.
    *   **Alpha Blending:** Uses soft edges to blend the object seamlessly without jagged artifacts.

4.  **Robust Fallback System:**
    *   If 3D tracking fails (e.g., due to blur or occlusion), the system gracefully falls back to a stable **Axis-Aligned Bounding Box (AABB)** method.
    *   *Result:* The object never disappears or glitches out.

### 🔊 Audio Preservation
- The final rendered video retains the **original high-quality audio** from the source video.
- Uses `ffmpeg` to merge the processed video stream with the original audio stream.

---

## 🛠️ Installation

### Prerequisites
- **OS:** Linux (Recommended) or Windows (WSL2)
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with CUDA support (Recommended for SAM3 performance)
- **FFmpeg:** Required for audio processing (`sudo apt install ffmpeg`)

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/prashant905/SAM3-Video-Segmentation-SOTA.git
    cd SAM3-Video-Segmentation-SOTA
    ```

2.  **Create Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Guide

### 1. Start the App
```bash
python app.py
```
Access the interface at: `http://localhost:7860`

### 2. Workflow
1.  **Upload Video:** Drag and drop your video file.
2.  **Select Object:**
    *   Choose "Point" or "Box" mode.
    *   Click on the object you want to replace.
3.  **Propagate Mask:**
    *   Click the orange **"Propagate across video"** button.
    *   Wait for the progress bar to finish.
4.  **Prepare Replacement:**
    *   Go to the "Gemini Image Generation" tab.
    *   Type a prompt (e.g., "A vintage Coca-Cola poster") or upload your own image.
5.  **Apply Edit:**
    *   Click **"🚀 Propagate AI Edit Across Video"**.
    *   The system will apply the SOTA pipeline (Homography + Harmonization) to every frame.
6.  **Export:**
    *   Click **"Render Video"**.
    *   Download the final MP4 file (Audio included!).

---

## ☁️ Server Deployment

To deploy this app on a remote server (e.g., AWS, GCP) and keep it running in the background:

1.  **Install tmux:**
    ```bash
    sudo apt-get update && sudo apt-get install -y tmux ffmpeg
    ```

2.  **Start a Persistent Session:**
    ```bash
    tmux new -s video_app
    ```

3.  **Run the App:**
    ```bash
    source venv/bin/activate
    python app.py
    ```

4.  **Detach:**
    *   Press `Ctrl` + `B`, then release and press `D`.
    *   The app will keep running. You can close your SSH terminal.

5.  **Re-attach (to check logs):**
    ```bash
    tmux attach -t video_app
    ```

---

## 🧩 Project Structure

- `app.py`: Main Gradio application containing the UI and logic.
- `requirements.txt`: Python dependencies.
- `README.md`: Documentation.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License.
