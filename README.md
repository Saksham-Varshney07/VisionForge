# VisionForge 🔍

A real-time product recognition system built with YOLOv8, Flask, and React. Train custom object detection models directly from your browser by uploading product videos — no coding required.

---

## What It Does

- 🎥 **Upload a video** of any product (e.g., a cereal box, shampoo bottle)
- 🤖 **Auto-annotates** frames and trains a YOLOv8 model on your custom product
- 📦 **Cumulative training** — each new product is added to the model without forgetting previous ones
- 📷 **Live recognition** via webcam with bounding boxes and confidence scores
- ⚡ **GPU-accelerated** training with CUDA support

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Backend API | Flask + Flask-SocketIO |
| Frontend | React + TypeScript + Vite |
| Styling | Tailwind CSS |
| Computer Vision | OpenCV |
| Real-time Comms | WebSocket (Socket.IO) |

---

## Project Structure

```
antiv1/
├── backend/
│   ├── app.py                  # Flask API & WebSocket server
│   ├── core/
│   │   ├── annotation.py       # Auto-annotation logic
│   │   ├── video_processor.py  # Frame extraction
│   │   ├── model_manager.py    # YOLOv8 training & YAML generation
│   │   └── __init__.py
│   ├── data/
│   │   ├── products_registry.json  # Tracks trained product classes
│   │   ├── datasets/               # Per-product annotated frames
│   │   └── master_dataset/         # Combined dataset for training
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.tsx             # Main UI (Dashboard + Pipeline tabs)
    │   └── index.css           # Design system
    └── package.json
```

---

## Getting Started

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## How to Train a Product

1. Go to the **Pipeline** tab
2. Enter the product name and upload a short video of it
3. Set the number of epochs (20–30 recommended)
4. Click **Start Pipeline** — it will extract frames, auto-annotate, and train
5. Switch to the **Dashboard** tab → click **Load Custom Model**
6. Hold the product in front of your webcam — it will be detected in real-time

> Each new product you train is **automatically added** to the model. After training 3 products, all 3 will be detected simultaneously.

---

## Notes

- `yolov8n.pt` (base model) is downloaded automatically on first run
- Trained weights are saved to `backend/data/models/custom_model/weights/best.pt`
- GPU is used automatically if CUDA is available; falls back to CPU otherwise
