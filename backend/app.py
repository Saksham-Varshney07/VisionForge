# Removed Eventlet to stabilize multipart file uploads

import os
import json
import shutil
import base64
import cv2
import numpy as np
import threading
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from core import extract_frames, auto_annotate_video, create_yolo_yaml, train_yolo_model, format_dataset_directory


# Initialize Flask and SocketIO
app = Flask(__name__)
# Allow CORS for development
CORS(app)
# Use standard threading since eventlet crashes PyTorch C++ extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
MODELS_DIR = os.path.join(DATA_DIR, "models")

for d in [VIDEOS_DIR, FRAMES_DIR, DATASETS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Global model loaded
current_model = None
current_model_type = 'default'  # 'default' or 'custom'
# Global lock to synchronize model loading and prevent CUDA initialization crashes
model_lock = threading.Lock()

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "online",
        "message": "VisionForge API is running",
        "engine": "YOLOv8",
        "model_loaded": current_model is not None
    }), 200

# Endpoint to upload video
@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(VIDEOS_DIR, filename)
        file.save(path)
        return jsonify({"message": "Video uploaded successfully", "path": path}), 200

# Endpoint to extract frames
@app.route('/api/extract', methods=['POST'])
def extract():
    data = request.json
    video_name = data.get('video_path')
    target_frames = data.get('target_frames', 60)
    
    # Resolve absolute path from filename
    video_path = os.path.join(VIDEOS_DIR, video_name)
    
    out_dir = os.path.join(FRAMES_DIR, os.path.basename(video_name).split('.')[0])
    success, msg = extract_frames(video_path, out_dir, target_frames)
    
    return jsonify({"success": success, "message": msg, "output_dir": out_dir}), 200 if success else 400

# Endpoint to auto annotate
@app.route('/api/annotate/auto', methods=['POST'])
def auto_annotate():
    data = request.json
    video_name = data.get('video_path')
    class_id = data.get('class_id', 0)
    video_path = os.path.join(VIDEOS_DIR, video_name)
    out_dir = os.path.join(DATASETS_DIR, os.path.basename(video_name).split('.')[0])
    success, msg = auto_annotate_video(video_path, out_dir, class_id)
    return jsonify({"success": success, "message": msg, "output_dir": out_dir}), 200 if success else 400

# Get the first frame of an uploaded video for manual box drawing
@app.route('/api/get_frame', methods=['POST'])
def get_first_frame():
    data = request.json
    video_name = data.get('video_path')
    video_path = os.path.join(VIDEOS_DIR, video_name)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"success": False, "message": "Could not read video frame"}), 400
    
    # Encode frame as JPEG base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    h, w = frame.shape[:2]
    
    return jsonify({"success": True, "frame": frame_b64, "width": w, "height": h})

# Endpoint to annotate using a user-drawn bounding box + OpenCV tracker
@app.route('/api/annotate/manual', methods=['POST'])
def manual_annotate():
    """
    Takes user-drawn box (normalized x,y,w,h on FIRST frame) and uses
    an OpenCV CSRT tracker to follow it across all extracted frames.
    This guarantees correct, tight bounding boxes on the actual product.
    """
    data = request.json
    video_name = data.get('video_path')
    class_id = data.get('class_id', 0)
    # Normalized coordinates from frontend canvas (0.0 to 1.0)
    norm_x = float(data.get('x', 0))
    norm_y = float(data.get('y', 0))
    norm_w = float(data.get('w', 0))
    norm_h = float(data.get('h', 0))
    
    video_path = os.path.join(VIDEOS_DIR, video_name)
    dataset_name = os.path.basename(video_name).split('.')[0]
    out_dir = os.path.join(DATASETS_DIR, dataset_name)
    
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"success": False, "message": "Cannot open video"}), 400
    
    ret, first_frame = cap.read()
    if not ret:
        return jsonify({"success": False, "message": "Cannot read first frame"}), 400
    
    fh, fw = first_frame.shape[:2]
    
    # Convert normalized coords to absolute pixel coords for the tracker
    px = int(norm_x * fw)
    py = int(norm_y * fh)
    pw = int(norm_w * fw)
    ph = int(norm_h * fh)
    
    # ORB-based feature tracking (works in standard opencv-python without contrib)
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Extract features from the template (the user-drawn box on the first frame)
    template = first_frame[py:py+ph, px:px+pw]
    template_kp, template_desc = orb.detectAndCompute(template, None)
    
    saved_count = 0
    frame_count = 0
    last_box = (px, py, pw, ph)  # Track last known position
    
    # Save first frame with initial box
    frame_path = os.path.join(out_dir, f"img_{saved_count:04d}.jpg")
    label_path = os.path.join(out_dir, f"img_{saved_count:04d}.txt")
    cv2.imwrite(frame_path, first_frame)
    cx = (px + pw/2) / fw
    cy = (py + ph/2) / fh
    nw = pw / fw
    nh = ph / fh
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    saved_count += 1
    
    # Track through the rest of the video, sampling every 5th frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if frame_count % 5 != 0:
            continue
        
        # Try ORB matching to find where the product moved to
        tracked_box = None
        if template_desc is not None and len(template_desc) > 10:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_kp, frame_desc = orb.detectAndCompute(gray, None)
                if frame_desc is not None and len(frame_desc) > 10:
                    matches = bf.match(template_desc, frame_desc)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = matches[:int(len(matches) * 0.3)]
                    
                    if len(good) > 8:
                        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                        
                        # Offset src_pts to original frame coordinates
                        src_offset = np.float32([[px, py]])
                        src_pts_abs = src_pts.reshape(-1, 2) + src_offset
                        dst_pts_flat = dst_pts.reshape(-1, 2)
                        
                        # Estimate translation using median shift (robust to outliers)
                        dx = float(np.median(dst_pts_flat[:, 0] - src_pts_abs[:, 0]))
                        dy = float(np.median(dst_pts_flat[:, 1] - src_pts_abs[:, 1]))
                        
                        lx, ly, lw, lh = last_box
                        tx = int(max(0, lx + dx))
                        ty = int(max(0, ly + dy))
                        tw = min(lw, fw - tx)
                        th = min(lh, fh - ty)
                        
                        if tw > 20 and th > 20:
                            tracked_box = (tx, ty, tw, th)
            except Exception:
                pass
        
        # Fall back to last known position if ORB matching fails
        final_box = tracked_box if tracked_box else last_box
        last_box = final_box
        
        tx, ty, tw, th = final_box
        cx = (tx + tw/2) / fw
        cy = (ty + th/2) / fh
        nw = tw / fw
        nh = th / fh
        
        frame_path = os.path.join(out_dir, f"img_{saved_count:04d}.jpg")
        label_path = os.path.join(out_dir, f"img_{saved_count:04d}.txt")
        cv2.imwrite(frame_path, frame)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        saved_count += 1
    
    cap.release()
    return jsonify({"success": True, "message": f"Annotated {saved_count} frames using tracker", "output_dir": out_dir})

# Load model
@app.route('/api/model/load', methods=['POST'])
def load_model():
    """
    Load a YOLO model into memory. We do this in a thread to prevent blocking 
    the Flask API (which was causing the frontend button to hang).
    """
    global current_model
    data = request.json
    model_name = data.get('model_name', 'yolov8n.pt')
    
    # We use a mutable container to capture thread results
    result = {"success": False, "message": ""}
    
    def _load():
        global current_model, current_model_type
        with model_lock:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                
                if model_name.startswith('custom_model'):
                    model_path = os.path.join(base_dir, 'data', 'models', 'custom_model', 'weights', 'best.pt')
                else:
                    model_path = os.path.join(base_dir, model_name)
                    
                print(f"[THREAD] Loading model from {model_path}...")
                new_model = YOLO(model_path)
                
                # Force inference to use CPU to prevent threading clashes between Flask and CUDA
                new_model.to('cpu')
                # Skip warmup here - it burns timeout budget and first real inference handles it
                
                current_model = new_model
                current_model_type = 'custom' if model_name.startswith('custom_model') else 'default'
                print(f"[THREAD] Model {model_name} loaded successfully! Type: {current_model_type}")
                result["success"] = True
                result["message"] = f"Loaded model {model_name}"
            except Exception as e:
                print(f"[THREAD] Model load failed: {e}")
                result["message"] = str(e)

    # Run in background to not block the Werkzeug worker
    loader_thread = threading.Thread(target=_load)
    loader_thread.start()
    loader_thread.join(timeout=60.0) # Wait up to 60 seconds for large custom models

    # If thread is still running after timeout, the model is still loading but we don't error
    if not result["success"] and result["message"] == "":
        return jsonify({"success": False, "message": "Model load timed out. Try again."})

    return jsonify({"success": result["success"], "message": result["message"]})

# Endpoint to train model — CUMULATIVE: each new product is added to the master dataset
@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    video_name = data.get('video_path')
    product_name = data.get('product_name', 'product').strip().lower().replace(' ', '_')
    epochs = data.get('epochs', 5)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    master_dir = os.path.join(base_dir, 'data', 'master_dataset')
    master_images = os.path.join(master_dir, 'images', 'train')
    master_labels = os.path.join(master_dir, 'labels', 'train')
    registry_path = os.path.join(base_dir, 'data', 'products_registry.json')

    os.makedirs(master_images, exist_ok=True)
    os.makedirs(master_labels, exist_ok=True)

    # Load or initialize the products registry {product_name: class_id}
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {}

    # Assign class ID — existing products keep their ID, new ones get the next one
    if product_name not in registry:
        registry[product_name] = len(registry)
        print(f"[TRAIN] New product '{product_name}' registered as class {registry[product_name]}")
    else:
        print(f"[TRAIN] Existing product '{product_name}' (class {registry[product_name]}) — updating its frames")

    class_id = registry[product_name]

    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    # Source: the newly annotated frames for this product
    source_dir = os.path.join(DATASETS_DIR, os.path.basename(video_name).split('.')[0])

    if not os.path.exists(source_dir):
        return jsonify({"success": False, "message": f"No annotated dataset found for {video_name}. Run annotation first."}), 400

    source_imgs = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.png')])
    source_lbls = sorted([f for f in os.listdir(source_dir) if f.endswith('.txt')])

    # Figure out offset so we don't overwrite existing master frames
    existing = [f for f in os.listdir(master_images) if f.startswith(f'cls{class_id}_')]
    # Remove old frames for this class (re-training same product refreshes its data)
    for old_img in existing:
        stem = os.path.splitext(old_img)[0]
        old_img_path = os.path.join(master_images, old_img)
        old_lbl_path = os.path.join(master_labels, f"{stem}.txt")
        if os.path.exists(old_img_path): os.remove(old_img_path)
        if os.path.exists(old_lbl_path): os.remove(old_lbl_path)

    # Copy new frames into master dataset with class-specific prefix and rewrite class ID
    for i, (img_file, lbl_file) in enumerate(zip(source_imgs, source_lbls)):
        new_stem = f"cls{class_id}_{i:04d}"

        # Copy image
        shutil.copy(os.path.join(source_dir, img_file), os.path.join(master_images, f"{new_stem}.jpg"))

        # Rewrite label with correct class_id
        src_lbl_path = os.path.join(source_dir, lbl_file)
        with open(src_lbl_path) as f:
            lines = f.readlines()
        with open(os.path.join(master_labels, f"{new_stem}.txt"), 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(class_id)  # Overwrite class ID
                    f.write(' '.join(parts) + '\n')

    total_master_images = len([f for f in os.listdir(master_images) if f.endswith('.jpg')])
    print(f"[TRAIN] Master dataset now has {total_master_images} images across {len(registry)} class(es): {list(registry.keys())}")

    # Build the class names list ordered by class ID
    sorted_classes = sorted(registry.items(), key=lambda x: x[1])
    class_names = [name for name, _ in sorted_classes]

    # Create data.yaml for the master dataset
    yaml_path = create_yolo_yaml(master_dir, class_names)

    # Train on the full master dataset
    success, res = train_yolo_model(yaml_path, epochs=epochs, project_dir=MODELS_DIR)

    if success:
        return jsonify({"success": True, "message": f"Training complete! Model now recognizes {len(registry)} product(s): {', '.join(class_names)}"})
    else:
        return jsonify({"success": False, "message": res})

@app.route('/api/scan_manual', methods=['POST'])
def scan_manual():
    global current_model
    if current_model is None:
        return jsonify({"success": False, "message": "No model loaded"}), 400
        
    data = request.json
    image_data = data.get('image')
    box = data.get('box')
    
    if not image_data or not box:
        return jsonify({"success": False, "message": "Missing image or box data"}), 400
        
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        b64_decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(b64_decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        h_img, w_img = img.shape[:2]
        x1 = max(0, int(box.get('x', 0)))
        y1 = max(0, int(box.get('y', 0)))
        w = max(0, int(box.get('w', 0)))
        h = max(0, int(box.get('h', 0)))
        x2 = min(w_img, x1 + w)
        y2 = min(h_img, y1 + h)
        
        if x2 <= x1 or y2 <= y1:
            return jsonify({"success": False, "message": "Invalid bounding box"}), 400
            
        cropped_img = img[y1:y2, x1:x2]
        
        with model_lock:
            # Use lower threshold to be forgiving on tight crops
            results = current_model(cropped_img, verbose=False, device='cpu', conf=0.05)
            
        best_det = None
        best_conf = -1
        
        for r in results:
            for b in r.boxes:
                conf = float(b.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    cls = int(b.cls[0])
                    best_det = current_model.names[cls]
                    
        if best_det and best_conf >= 0.1: # At least 10% confident for manual tightly cropped boxes
            return jsonify({
                "success": True, 
                "class": best_det, 
                "confidence": best_conf
            }), 200
        else:
            return jsonify({"success": False, "message": "No confident detection found in area."}), 200
            
    except Exception as e:
        return jsonify({"success": False, "message": f"Scan failed: {str(e)}"}), 500

@socketio.on('connect')
def test_connect():
    print("Socket connected!")
    emit('status', {'data': 'Connected to Inference Engine'})

@socketio.on('stream_frame')
def handle_frame(data):
    """
    data['image']: base64 encoded image
    """
    global current_model
    
    if current_model is None:
        emit('detections', {'error': 'No model loaded'})
        return
        
    try:
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        b64_decoded = base64.b64decode(image_data)
        np_arr = np.frombuffer(b64_decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with model_lock:
            if current_model is None:
                return
            # Use lower confidence for custom model (small dataset = lower scores)
            # Default model uses standard 0.25 threshold
            conf_threshold = 0.15 if current_model_type == 'custom' else 0.25
            # Force device to CPU for inference to avoid CUDA thread locks
            results = current_model(img, verbose=False, device='cpu', conf=conf_threshold)
            
        boxes = []
        for r in results:
            for box in r.boxes:
                # box format: xyxy
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = current_model.names[cls]
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf, "class": name
                })
        
        emit('detections', {"detections": boxes})
    except Exception as e:
        emit('detections', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
