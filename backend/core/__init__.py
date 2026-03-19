# Expose module functions
from .video_processor import extract_frames
from .annotation import auto_annotate_video
from .model_manager import create_yolo_yaml, train_yolo_model, format_dataset_directory
