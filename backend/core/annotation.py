import cv2
import os
import numpy as np
from ultralytics import YOLO

def auto_annotate_video(video_path, output_dir, class_id=0):
    """
    Given a video (even with moving camera), use a pre-trained YOLOv8n to bootstrap bounding boxes.
    It will detect any object and we treat the largest non-person object as our product class.
    If YOLO fails to find anything, falls back to GrabCut to find the largest foreground object.
    As a final fallback, it uses a wide central bounding box.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video"

    # Use the base yolov8n model for generic objectness detection
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'yolov8n.pt')
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        return False, f"Failed to load fallback model: {e}"
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 5th frame to get diverse samples
        if count % 5 == 0:
            best_box = None # Normalized cx, cy, w, h
            img_h, img_w = frame.shape[:2]
            
            # --- Attempt 1: YOLOv8n detection for non-person objects ---
            try:
                # Force device to CPU during annotation to prevent CUDA Context crashes in background threads
                results = model(frame, verbose=False, device='cpu')
                max_area = 0
                yolo_bbox_xyxy = None
                
                for r in results:
                    for box in r.boxes:
                        # Ignore 'person' class (index 0) to avoid picking up hands
                        if int(box.cls[0]) == 0:
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        
                        # Consider objects that are not too small and not almost the entire frame
                        if area > (img_w * img_h * 0.01) and area < (img_w * img_h * 0.9): 
                            if area > max_area:
                                max_area = area
                                yolo_bbox_xyxy = [x1, y1, x2, y2]
                
                if yolo_bbox_xyxy:
                    x1, y1, x2, y2 = yolo_bbox_xyxy
                    
                    # Expand the bounding box by 10% on all sides
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    expand_factor = 0.1
                    new_width = width * (1 + expand_factor)
                    new_height = height * (1 + expand_factor)
                    
                    # Calculate new coordinates, ensuring they stay within image bounds
                    new_x1 = max(0, center_x - new_width / 2)
                    new_y1 = max(0, center_y - new_height / 2)
                    new_x2 = min(img_w, center_x + new_width / 2)
                    new_y2 = min(img_h, center_y + new_height / 2)
                    
                    # Convert to normalized cx, cy, w, h
                    best_box = [
                        ((new_x1 + new_x2) / 2) / img_w,
                        ((new_y1 + new_y2) / 2) / img_h,
                        (new_x2 - new_x1) / img_w,
                        (new_y2 - new_y1) / img_h
                    ]
            except Exception as e:
                print(f"YOLO detection failed: {e}")
                pass # Continue to next fallback if YOLO fails
                    
            # --- Attempt 2: GrabCut fallback if YOLO finds nothing or fails ---
            if not best_box:
                try:
                    # Scale down the image for much faster GrabCut processing
                    MAX_DIM = 256
                    scale_factor = 1.0
                    
                    bg_frame = frame
                    if img_w > MAX_DIM or img_h > MAX_DIM:
                        scale_factor = MAX_DIM / max(img_w, img_h)
                        bg_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
                        
                    s_h, s_w = bg_frame.shape[:2]
                    mask = np.zeros(bg_frame.shape[:2], np.uint8)
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)

                    # Central working area (20% padding around edges)
                    s_rect = (int(s_w * 0.2), int(s_h * 0.2), int(s_w * 0.6), int(s_h * 0.6))
                    
                    cv2.grabCut(bg_frame, mask, s_rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
                    # 2 and 0 are background/probable background in GrabCut. 1 and 3 are foreground.
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    
                    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        # Scale area threshold dynamically
                        if cv2.contourArea(c) > (MAX_DIM * MAX_DIM * 0.05):
                            x, y, w, h = cv2.boundingRect(c)
                            # Convert back to normalized coordinates of the ORIGINAL frame
                            x_center = (x + w/2) / s_w
                            y_center = (y + h/2) / s_h
                            n_w = w / s_w
                            n_h = h / s_h
                            if (n_w * n_h) < 0.9: # avoid capturing the entire screen
                                best_box = [x_center, y_center, n_w, n_h]
                except Exception as e:
                    print(f"Grabcut failed: {e}")
                    pass
                    
            if not best_box:
                # Absolute rigid fallback if computer vision completely fails
                best_box = [0.5, 0.5, 0.4, 0.5]

            if best_box:
                x_center, y_center, n_w, n_h = best_box
                label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {n_w:.6f} {n_h:.6f}"
                
                frame_path = os.path.join(output_dir, f"{base_name}_auto_{saved_count:04d}.jpg")
                label_path = os.path.join(output_dir, f"{base_name}_auto_{saved_count:04d}.txt")
                
                cv2.imwrite(frame_path, frame)
                with open(label_path, 'w') as f:
                    f.write(label_line)
                    
                saved_count += 1
                
        count += 1
        
    cap.release()
    return True, f"Auto-annotated {saved_count} frames using YOLO bootstrap"
