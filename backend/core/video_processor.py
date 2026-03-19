import cv2
import os

def extract_frames(video_path, output_dir, target_frames=60):
    """
    Extract a target number of frames evenly across a video.
    """
    if not os.path.exists(video_path):
        return False, "Video file not found"
    
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Failed to open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or total_frames_in_video <= 0:
        # Fallback if properties are missing
        fps = 30
        total_frames_in_video = 300
        
    # Calculate how many frames to skip to get exactly target_frames
    frame_interval = max(1, int(total_frames_in_video / target_frames))

    count = 0
    saved_count = 0
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{base_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    return True, f"Extracted {saved_count} frames to {output_dir}"
