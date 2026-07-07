import os
import sys
import time
import hashlib
import collections
import cv2
import torch
import numpy as np
from ultralytics import YOLOE
from ultralytics.utils import SETTINGS

# Define the custom classes you want the model to detect and label.
# You can modify this list as needed.
CUSTOM_CLASSES = ["person", "cup", "chair", "cell phone", "bottle", "keyboard", "mouse"]

# Setup paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
EMBEDDINGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../embeddings"))

# Configure Ultralytics to search for model weights (like CLIP) in the models folder
SETTINGS.update({"weights_dir": MODELS_DIR})

def draw_semi_transparent_rect(img, pt1, pt2, color, alpha):
    """Draw a semi-transparent rectangle on the image."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        sub_img = img[y1:y2, x1:x2]
        rect = np.full(sub_img.shape, color, dtype=np.uint8)
        img[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)

def get_embeddings_path(classes):
    """Generate a deterministic and unique filename for a given list of classes."""
    classes_str = ",".join(sorted(classes))
    h = hashlib.md5(classes_str.encode("utf-8")).hexdigest()
    return os.path.join(EMBEDDINGS_DIR, f"yoloe_emb_{h}.pt")

def main():
    print("Checking torch MPS availability...")
    mps_available = torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_available(): {mps_available}")
    
    if not mps_available:
        print("ERROR: MPS (Metal Performance Shaders) is not available.")
        print("This script is configured to run on MPS for performance validation.")
        device = "cpu"
    else:
        print("SUCCESS: MPS is available.")
        device = "mps"

    # Make sure folders exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "yoloe-26s-seg.pt")
    print(f"Loading standard YOLOE model from {model_path}...")
    
    # Check if the model file exists, if not it will automatically download it to MODELS_DIR
    model = YOLOE(model_path)
    print("Model loaded successfully.")

    # Determine unique path for this exact combination of custom classes
    emb_path = get_embeddings_path(CUSTOM_CLASSES)
    
    if os.path.exists(emb_path):
        print(f"Found cached embeddings at: {emb_path}")
        print("Loading cached embeddings...")
        embeddings = torch.load(emb_path, weights_only=True).to(device)
        model.set_classes(CUSTOM_CLASSES, embeddings=embeddings)
        print("Cached custom classes set successfully.")
    else:
        print(f"No cached embeddings found for classes: {CUSTOM_CLASSES}")
        print("Generating text embeddings using CLIP (this may take a moment)...")
        # Generate text embeddings
        embeddings = model.get_text_pe(CUSTOM_CLASSES)
        # Save to cache folder (on CPU for portability)
        torch.save(embeddings.cpu(), emb_path)
        print(f"Saved generated embeddings to cache: {emb_path}")
        
        # Apply to model
        model.set_classes(CUSTOM_CLASSES, embeddings=embeddings.to(device))
        print("Generated custom classes set successfully.")

    print("Opening default webcam (device 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Webcam successfully opened. Press 'q' in the window to quit.")
    
    # Deque to keep track of elapsed time for the last 30 frames
    frame_times = collections.deque(maxlen=30)
    prev_time = time.perf_counter()
    
    # BGR color for Cyan (HUD style)
    hud_color = (255, 255, 0)

    # Dictionary to store active tracking IDs from the previous frame: track_id -> class_name
    active_tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run YOLOE tracking with device='mps' (or fallback)
        # We specify conf=0.4 to filter out low-confidence detections early
        # persist=True maintains tracking state between frames
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            device=device, 
            conf=0.4, 
            verbose=False
        )
        
        current_tracks = {}

        # Process detections & tracks
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None
            track_ids = boxes.id.int().cpu().tolist() if has_ids else []
            
            for i, box in enumerate(boxes):
                # Bounding box coordinates (xyxy)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Confidence score
                conf = float(box.conf[0])
                
                # Class index & name
                cls_id = int(box.cls[0])
                class_name = model.names.get(cls_id, f"class_{cls_id}")
                
                # Draw bounding box (thin line)
                cv2.rectangle(frame, (x1, y1), (x2, y2), hud_color, 1)
                
                # Get tracking ID if available
                if has_ids and i < len(track_ids):
                    track_id = track_ids[i]
                    current_tracks[track_id] = class_name
                    label_text = f"ID {track_id} | {class_name} {conf:.2f}"
                else:
                    label_text = f"{class_name} {conf:.2f}"
                
                # Get text size for background box
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                
                # Label positioning: above the box, or inside if too close to the top
                tx = x1
                ty = y1 - 4
                if ty - text_h < 0:
                    ty = y1 + text_h + 4
                
                # Draw semi-transparent background behind text for readability
                bg_pt1 = (tx, ty - text_h - 2)
                bg_pt2 = (tx + text_w + 4, ty + baseline)
                draw_semi_transparent_rect(frame, bg_pt1, bg_pt2, (0, 0, 0), 0.6)
                
                # Draw text label
                cv2.putText(
                    frame,
                    label_text,
                    (tx + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Log to the console whenever a track ID appears or disappears
        new_ids = set(current_tracks.keys()) - set(active_tracks.keys())
        disappeared_ids = set(active_tracks.keys()) - set(current_tracks.keys())
        
        for tid in new_ids:
            print(f"[TRACKER] New track ID appeared: ID {tid} ({current_tracks[tid]})")
            
        for tid in disappeared_ids:
            print(f"[TRACKER] Track ID disappeared: ID {tid} ({active_tracks[tid]})")
            
        active_tracks = current_tracks

        # Calculate Running FPS (averaged over the last 30 frames)
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time
        frame_times.append(dt)
        
        fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw FPS in the corner of the frame (top-left) with a semi-transparent background
        (fps_w, fps_h), fps_baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        fps_bg_pt1 = (10, 10)
        fps_bg_pt2 = (10 + fps_w + 10, 10 + fps_h + fps_baseline + 8)
        draw_semi_transparent_rect(frame, fps_bg_pt1, fps_bg_pt2, (0, 0, 0), 0.6)
        
        cv2.putText(
            frame,
            fps_text,
            (15, 10 + fps_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Display output frame
        cv2.imshow("YOLOE Custom Prompt Overlay", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference loop finished. Webcam released.")

if __name__ == "__main__":
    main()
