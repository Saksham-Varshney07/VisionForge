import os
import yaml
import sys
import subprocess
from ultralytics import YOLO

def create_yolo_yaml(dataset_dir, class_names):
    """
    Creates a data.yaml file needed by YOLOv8 for training.
    Assumes dataset_dir has 'images/train' and 'labels/train' structure.
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    # We use absolute paths or relative to the dataset config for Ultralytics
    # The safest is absolute paths
    abs_dataset_dir = os.path.abspath(dataset_dir)
    
    data = {
        'path': abs_dataset_dir,
        'train': 'images/train',
        'val': 'images/train', # Use same for simple demo, ideally separate val set
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
    return yaml_path

def train_yolo_model(data_yaml_path, epochs=10, model_name="yolov8n.pt", project_dir="run", name="custom_model"):
    """
    Trains a YOLOv8 model using the provided data.yaml via an isolated subprocess.
    This bypasses Flask thread context which crashes CUDA initialization on Windows.
    """
    try:
        # Define an isolated script to run PyTorch without Flask worker conflicts
        script = f'''
from ultralytics import YOLO
import sys

try:
    model = YOLO('{model_name}')
    model.train(
        data=r'{data_yaml_path}',
        epochs={epochs},
        project=r'{project_dir}',
        name='{name}',
        exist_ok=True,
        workers=0,
        device=0
    )
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
'''
        # Execute perfectly isolated training process but pipe output directly to the terminal
        process = subprocess.Popen([sys.executable, "-c", script], stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        
        if process.returncode == 0:
            return True, "Training finished successfully!"
        else:
            return False, "Training process failed. Check terminal for errors."
            
    except Exception as e:
        return False, str(e)

def format_dataset_directory(base_dir, source_frames_dir, source_labels_dir):
    """
    Formats the flat directory into images/train and labels/train.
    """
    import shutil
    
    images_dir = os.path.join(base_dir, "images", "train")
    labels_dir = os.path.join(base_dir, "labels", "train")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # YOLOv8 requires that if an image is 'images/train/frame_0001.jpg' 
    # its label must be exactly 'labels/train/frame_0001.txt'.
    # Since our extractor/annotator saves them as `name_frame_0001.jpg` and `name_auto_0001.txt` 
    # we need to normalize the names when copying to the final dataset directory
    
    # 1. Collect all frames
    frames = sorted([f for f in os.listdir(source_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
    labels = sorted([f for f in os.listdir(source_labels_dir) if f.endswith('.txt')])
    
    # Simple copy over (In a real app, split train/test)
    for i, frame_filename in enumerate(frames):
        # We rename them to simple numeric IDs for YOLO to match them perfectly
        new_basename = f"img_{i:04d}"
        
        # Copy image
        shutil.copy(
            os.path.join(source_frames_dir, frame_filename), 
            os.path.join(images_dir, f"{new_basename}.jpg")
        )
        
        # In our current setup, annotations are saved separately. To match them,
        # we try to find the corresponding label if it exists.
        # Since the annotator currently saves frames AND labels, we know they align.
        if i < len(labels):
             shutil.copy(
                 os.path.join(source_labels_dir, labels[i]), 
                 os.path.join(labels_dir, f"{new_basename}.txt")
             )
            
    return base_dir
