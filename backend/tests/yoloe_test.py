import os
import sys
import time
import collections
import queue
import cv2
import torch
import numpy as np
from ultralytics import YOLOE
import threading
import json
import base64
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import db

def cv2_to_base64_data_url(img):
    """Encode an OpenCV image (numpy array) to a base64 Data URL."""
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def execute_tagging_with_retries(client, messages, groq_client=None):
    """Executes tagging. First tries nvidia/llama-3.1-nemotron-nano-vl-8b-v1 as the primary VLM.
    Falls back to qwen/qwen3.5-397b-a17b (NVIDIA).
    Falls back to qwen/qwen3.6-27b (Groq).
    As a fourth option, falls back to nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (NVIDIA).
    """
    def parse_response(content):
        if not content:
            raise ValueError("Received empty content from API")
        content_clean = content.strip()
        if content_clean.startswith("```json"):
            content_clean = content_clean[7:]
        elif content_clean.startswith("```"):
            content_clean = content_clean[3:]
        if content_clean.endswith("```"):
            content_clean = content_clean[:-3]
        content_clean = content_clean.strip()
        
        result_data = json.loads(content_clean)
        if "object_name" not in result_data or "tags" not in result_data:
            raise KeyError("Missing required keys ('object_name', 'tags') in API response JSON")
        return result_data

    # 1. Attempt Primary VLM: Llama 3.1 Nemotron Nano VL (Extremely fast, reliable Vision model)
    print("[WORKER] Invoking primary VLM (nvidia/llama-3.1-nemotron-nano-vl-8b-v1)...")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
            ConnectionError,
            TimeoutError
        )),
        reraise=True
    )
    def call_llama_vl():
        return client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=messages,
            temperature=0.4,
            top_p=0.9,
            max_tokens=4096,
            timeout=20.0
        )

    for attempt in range(1, 3):
        try:
            response = call_llama_vl()
            content = response.choices[0].message.content
            result_data = parse_response(content)
            
            confidence = result_data.get("confidence", "low").lower()
            if confidence == "low":
                raise ValueError("Primary VLM (Llama-VL) returned low confidence")
                
            print("[WORKER] Primary VLM (Llama-VL) tagging successful.")
            return result_data
        except Exception as e:
            if attempt < 2:
                print(f"[WORKER] Llama-VL attempt {attempt} failed or returned low confidence: {e}. Retrying once...")
                continue
            else:
                print(f"[WORKER] Primary VLM (nvidia/llama-3.1-nemotron-nano-vl-8b-v1) failed or timed out: {e}. Trying Qwen...")

    # 2. Fallback VLM 1: Qwen 3.5 397B
    print("[WORKER] Invoking fallback VLM 1 (qwen/qwen3.5-397b-a17b)...")
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.5-397b-a17b",
            messages=messages,
            max_tokens=16384,
            temperature=0.60,
            top_p=0.95,
            presence_penalty=0,
            extra_body={
                "top_k": 20,
                "repetition_penalty": 1
            },
            timeout=60.0
        )
        content = response.choices[0].message.content
        result_data = parse_response(content)
        
        confidence = result_data.get("confidence", "low").lower()
        if confidence == "low":
            raise ValueError("Fallback VLM 1 (Qwen) returned low confidence")
            
        print("[WORKER] Fallback VLM 1 (Qwen) tagging successful.")
        return result_data
        
    except Exception as err:
        print(f"[WORKER] Fallback VLM 1 (qwen/qwen3.5-397b-a17b) failed or timed out: {err}. Trying Groq Qwen...")

    # 3. Fallback VLM 2: Groq Qwen 3.6 27b
    if groq_client is None:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_api_key
            )

    if groq_client is not None:
        print("[WORKER] Invoking fallback VLM 2 (qwen/qwen3.6-27b via Groq)...")
        try:
            response = groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=messages,
                max_tokens=4096,
                temperature=0.60,
                top_p=0.95,
                timeout=60.0
            )
            content = response.choices[0].message.content
            result_data = parse_response(content)
            
            confidence = result_data.get("confidence", "low").lower()
            if confidence == "low":
                raise ValueError("Fallback VLM 2 (Groq Qwen) returned low confidence")
                
            print("[WORKER] Fallback VLM 2 (Groq Qwen) tagging successful.")
            return result_data
            
        except Exception as err:
            print(f"[WORKER] Fallback VLM 2 (qwen/qwen3.6-27b via Groq) failed or timed out: {err}. Trying Nemotron Reasoning...")
    else:
        print("[WORKER] Skipping fallback VLM 2 (Groq Qwen) because GROQ_API_KEY is not set. Trying Nemotron Reasoning...")

    # 4. Fallback VLM 3: Nemotron 3 Nano Omni (with reasoning, tenacity retries)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
            ConnectionError,
            TimeoutError
        )),
        reraise=True
    )
    def call_fallback_api():
        return client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=65536,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
            timeout=180.0
        )

    print("[WORKER] Invoking fallback VLM 3 (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)...")
    for attempt in range(1, 3):
        try:
            response = call_fallback_api()
            content = response.choices[0].message.content
            result_data = parse_response(content)
            
            confidence = result_data.get("confidence", "low").lower()
            if confidence == "low":
                if attempt < 2:
                    print(f"[WORKER] Fallback VLM 3 returned low confidence on attempt 1. Retrying once...")
                    continue
                else:
                    raise ValueError("Low confidence response from fallback VLM 3 after retry")
                    
            print("[WORKER] Fallback VLM 3 (Nemotron Reasoning) tagging successful.")
            return result_data
            
        except ValueError as e:
            if attempt < 2:
                print(f"[WORKER] Fallback VLM 3 attempt 1 failed with value error: {e}. Retrying once...")
                continue
            else:
                raise e
        except Exception as e:
            # Fail immediately on API/network errors since tenacity already handled retries
            raise e

def tagging_worker_func(finalized_queue, model_path):
    """Background worker thread function.
    Loads YOLOE on CPU for embedding, instantiates OpenAI client,
    and processes tracks from finalized_queue with de-duplication and tagging.
    """
    print("[WORKER] Background worker thread started. Loading YOLOE model on CPU for embeddings...")
    embedding_model = YOLOE(model_path)
    
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[WORKER] WARNING: NVIDIA_API_KEY environment variable is not set. API calls will fail.")
        
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key or "missing_key"
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("[WORKER] WARNING: GROQ_API_KEY environment variable is not set. Groq fallback will be bypassed.")

    groq_client = None
    if groq_api_key:
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key
        )
    
    while True:
        item = finalized_queue.get()
        if item is None:
            print("[WORKER] Received stop signal. Exiting background worker...")
            finalized_queue.task_done()
            break
            
        track_id = item['track_id']
        first_seen = item['first_seen']
        last_seen = item['last_seen']
        crops = item['crops']
        
        try:
            # 1. Compute embedding for the representative crop
            # representative crop is crops[0]['image'] as crops are sorted descending by quality score
            representative_crop = crops[0]['image'] if crops else None
            if representative_crop is None:
                raise ValueError("No crops available for track")
                
            embs = embedding_model.embed(representative_crop, device='cpu', verbose=False)
            emb_tensor = embs[0]
            emb_np = emb_tensor.numpy().astype(np.float32)
            emb_bytes = emb_np.tobytes()
            
            # Use max confidence of all crops as tracking confidence
            confidence = max([c['confidence'] for c in crops]) if crops else 0.0
            
            # 2. De-duplication check against tagged objects in DB
            is_duplicate = False
            best_match = None
            highest_similarity = -1.0
            
            try:
                with db.get_db_connection() as conn:
                    rows = conn.execute(
                        "SELECT track_id, last_seen, crop_paths, tags, ocr_text, embedding FROM objects WHERE status = 'tagged' AND embedding IS NOT NULL"
                    ).fetchall()
                    
                for row in rows:
                    db_emb_bytes = row['embedding']
                    db_emb = np.frombuffer(db_emb_bytes, dtype=np.float32)
                    
                    # Cosine similarity
                    dot_product = np.dot(emb_np, db_emb)
                    norm_a = np.linalg.norm(emb_np)
                    norm_b = np.linalg.norm(db_emb)
                    similarity = dot_product / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = row
                
                # Check threshold
                if highest_similarity >= DEDUP_SIMILARITY_THRESHOLD and best_match is not None:
                    is_duplicate = True
            except Exception as db_err:
                print(f"[WORKER ERROR] Error querying database for de-duplication: {db_err}")
                
            # Log dedup decision to the console
            if best_match is not None:
                print(f"[DEDUP] Best match for track {track_id} is existing track {best_match['track_id']} with similarity {highest_similarity:.4f} (Threshold: {DEDUP_SIMILARITY_THRESHOLD})")
            else:
                print(f"[DEDUP] No match found for track {track_id} in database (Threshold: {DEDUP_SIMILARITY_THRESHOLD})")
                
            if is_duplicate:
                print(f"[DEDUP] Treat as re-sighting of existing track {best_match['track_id']}. Updating last_seen and crops.")
                db.update_object_re_sighting(best_match['track_id'], last_seen, crops)
                # No VLM API call!
                continue
                
            # 3. If not duplicate, proceed with the normal pending-insert-and-tag flow
            db.insert_pending_object(
                track_id=track_id,
                first_seen=first_seen,
                last_seen=last_seen,
                confidence=confidence,
                crops=crops
            )
            print(f"[STATUS] Track {track_id} transition: None -> pending")
            
            # 4. Construct Vision API call
            content_blocks = [
                {
                    "type": "text",
                    "text": (
                        "You are a computer vision assistant. You are given a sequence of cropped images of the same object tracked in a video feed. "
                        "Determine the name of the object, extract any printed text on it, and label it with descriptive tags.\n"
                        "You MUST return ONLY a strict JSON object with no explanations, no wrapping markdown formatting code blocks, and no extra text. "
                        "The JSON object must have EXACTLY the following keys:\n"
                        "- 'object_name': A short, clear name/type of the object (string).\n"
                        "- 'tags': A JSON array/list of short descriptive tags (strings, e.g. ['red', 'nylon', 'water bottle']).\n"
                        "- 'ocr_text': Any printed characters, logos, or text visible on the object (string, empty string if none found).\n"
                        "- 'confidence': Your confidence rating for this identification, which must be exactly one of: 'high', 'medium', or 'low'.\n"
                        "Example:\n"
                        '{"object_name": "keyboard", "tags": ["black", "mechanical", "plastic"], "ocr_text": "Logitech", "confidence": "high"}'
                    )
                }
            ]
            
            for crop in crops:
                data_url = cv2_to_base64_data_url(crop['image'])
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                })
                
            messages = [
                {
                    "role": "user",
                    "content": content_blocks
                }
            ]
            
            # Call API with retry logic
            result = execute_tagging_with_retries(client, messages, groq_client=groq_client)
            
            # Write back result (pending -> tagged), and store embedding once tagging succeeds
            db.update_object_result(
                track_id=track_id,
                tags=result['tags'],
                ocr_text=result.get('ocr_text', ''),
                status='tagged',
                embedding=emb_bytes
            )
            print(f"[STATUS] Track {track_id} transition: pending -> tagged")
            
        except Exception as err:
            print(f"[WORKER ERROR] Permanent tagging failure for track {track_id}: {err}")
            try:
                db.update_object_result(
                    track_id=track_id,
                    tags=None,
                    ocr_text=None,
                    status='failed'
                )
                print(f"[STATUS] Track {track_id} transition: pending -> failed")
            except Exception as db_err:
                print(f"[WORKER ERROR] Failed to update status to failed in database for track {track_id}: {db_err}")
                
        finally:
            finalized_queue.task_done()

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

# Threshold constants for crop filtering
MIN_SHARPNESS = 50.0   # Minimum Laplacian variance for a crop to be considered sharp
MIN_BBOX_SIZE = 1600   # Minimum bounding box area in pixels (width * height)
MIN_CONFIDENCE = 0.25  # Minimum detection confidence score
DEDUP_SIMILARITY_THRESHOLD = 0.9  # Threshold above which an object is considered a re-sighting

# Weights for the combined quality score (weighted combination of confidence, size, and sharpness)
WEIGHT_CONF = 1.0
WEIGHT_SIZE = 1.0
WEIGHT_SHARP = 1.0

def compute_blur_score(crop):
    """Compute the Laplacian variance as a sharpness/blur score.
    Higher values mean sharper images, lower values mean blurrier images.
    """
    if crop is None or crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

def compute_combined_score(confidence, bbox_size, sharpness):
    """Compute combined quality score using normalized scale weights."""
    scaled_size = bbox_size / 10000.0
    scaled_sharp = sharpness / 100.0
    return (WEIGHT_CONF * confidence) + (WEIGHT_SIZE * scaled_size) + (WEIGHT_SHARP * scaled_sharp)

class TrackBuffer:
    def __init__(self, max_crops=6, iou_similarity_thresh=0.7):
        self.max_crops = max_crops
        self.iou_similarity_thresh = iou_similarity_thresh
        # List of dicts: {'image': np.ndarray, 'bbox': list, 'confidence': float, 'sharpness': float, 'score': float}
        self.crops = []

    def add_crop(self, crop_img, bbox, confidence, sharpness, score):
        """Attempts to add a crop to the buffer.
        Returns:
            bool: True if crop was added or replaced an existing crop, False otherwise.
            str: Description of the action taken (e.g. 'added', 'replaced', 'rejected').
        """
        new_crop = {
            'image': crop_img.copy(),
            'bbox': list(bbox),
            'confidence': confidence,
            'sharpness': sharpness,
            'score': score
        }
        
        # Check if the new crop is spatially too similar to any existing crop in the buffer
        similar_crop_idx = -1
        for idx, crop in enumerate(self.crops):
            if compute_iou(bbox, crop['bbox']) > self.iou_similarity_thresh:
                similar_crop_idx = idx
                break
                
        if similar_crop_idx != -1:
            existing_crop = self.crops[similar_crop_idx]
            if score > existing_crop['score']:
                self.crops[similar_crop_idx] = new_crop
                self.crops.sort(key=lambda x: x['score'], reverse=True)
                return True, f"replaced similar crop (old score: {existing_crop['score']:.2f}, new score: {score:.2f})"
            else:
                return False, f"ignored (similar to existing with score {existing_crop['score']:.2f})"
        
        # If not spatially similar, check capacity
        if len(self.crops) < self.max_crops:
            self.crops.append(new_crop)
            self.crops.sort(key=lambda x: x['score'], reverse=True)
            return True, f"added (capacity: {len(self.crops)}/{self.max_crops})"
        
        # Buffer is full, check if it is better than the worst crop (last one)
        worst_crop = self.crops[-1]
        if score > worst_crop['score']:
            self.crops[-1] = new_crop
            self.crops.sort(key=lambda x: x['score'], reverse=True)
            return True, f"evicted worst (score: {worst_crop['score']:.2f}) and replaced with new (score: {score:.2f})"
            
        return False, f"rejected (score {score:.2f} <= worst score {worst_crop['score']:.2f})"

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

    print("Loading prompt-free YOLOE model (yoloe-26l-seg-pf.pt)...")
    # Load the YOLOE checkpoint using path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "../models/yoloe-26l-seg-pf.pt"))
    model = YOLOE(model_path)
    print(f"Model loaded successfully from {model_path}.")
    
    # Initialize SQLite database
    print("Initializing SQLite database...")
    db.init_db()
    
    # Simple in-memory queue for finalized accepted tracks
    finalized_queue = queue.Queue()
    
    # Start background tagging worker thread
    print("Starting background tagging worker thread...")
    worker_thread = threading.Thread(
        target=tagging_worker_func,
        args=(finalized_queue, model_path),
        daemon=True
    )
    worker_thread.start()
    
    # Path to custom ByteTrack configuration
    tracker_path = os.path.abspath(os.path.join(script_dir, "custom_bytetrack.yaml"))

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

    # In-memory buffer keyed by track ID
    track_buffers = {}

    # Dictionary to store active track metadata: track_id -> {first_seen, last_seen, class_counts}
    track_metadata = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Keep a clean copy of the frame for cropping (before overlays are drawn)
        frame_clean = frame.copy()

        # Calculate time delta for instantaneous FPS and rolling FPS
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time
        frame_times.append(dt)
        inst_fps = 1.0 / dt if dt > 0 else 0.0

        # Run YOLOE tracking with device='mps' (or fallback)
        # We specify the custom ByteTrack config file.
        # We specify conf=0.1 matching track_low_thresh so that low-confidence
        # detections are not pre-filtered and can be used for track association.
        # persist=True maintains tracking state between frames
        results = model.track(
            frame, 
            persist=True, 
            tracker=tracker_path, 
            device=device, 
            conf=0.1, 
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
                    
                    # Update track metadata
                    if track_id not in track_metadata:
                        track_metadata[track_id] = {
                            'first_seen': curr_time,
                            'last_seen': curr_time,
                            'class_counts': collections.Counter({class_name: 1})
                        }
                    else:
                        track_metadata[track_id]['last_seen'] = curr_time
                        track_metadata[track_id]['class_counts'][class_name] += 1
                    
                    # Ensure track has a buffer
                    if track_id not in track_buffers:
                        track_buffers[track_id] = TrackBuffer()
                    
                    # Crop bbox safely from clean frame
                    h, w = frame_clean.shape[:2]
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w, x2), min(h, y2)
                    
                    if x2_c > x1_c and y2_c > y1_c:
                        crop_img = frame_clean[y1_c:y2_c, x1_c:x2_c]
                        bbox_size = (x2_c - x1_c) * (y2_c - y1_c)
                        
                        # Validate thresholds
                        if conf < MIN_CONFIDENCE:
                            pass
                        elif bbox_size < MIN_BBOX_SIZE:
                            pass
                        else:
                            sharpness = compute_blur_score(crop_img)
                            if sharpness >= MIN_SHARPNESS:
                                score = compute_combined_score(conf, bbox_size, sharpness)
                                added, action_desc = track_buffers[track_id].add_crop(
                                    crop_img, [x1_c, y1_c, x2_c, y2_c], conf, sharpness, score
                                )
                                print(f"[CROP BUFFER] Track {track_id} ({class_name}): {action_desc} | "
                                      f"Conf: {conf:.2f}, Size: {bbox_size}px, Sharpness: {sharpness:.1f}, Combined Score: {score:.2f}")
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
            print(f"[TRACKER] [FPS: {inst_fps:.1f}] New track ID appeared: ID {tid} ({current_tracks[tid]})")
            
        for tid in disappeared_ids:
            print(f"[TRACKER] [FPS: {inst_fps:.1f}] Track ID disappeared: ID {tid} ({active_tracks[tid]})")
            
        active_tracks = current_tracks
        
        # Log active buffers status summary
        if current_tracks:
            buffer_summary = []
            for tid in sorted(current_tracks.keys()):
                buf = track_buffers.get(tid)
                if buf:
                    scores_str = ", ".join([f"{c['score']:.2f}" for c in buf.crops])
                    buffer_summary.append(f"ID {tid}: {len(buf.crops)} crops [{scores_str}]")
            if buffer_summary:
                print(f"[BUFFER STATUS] { ' | '.join(buffer_summary) }")

        # Check for finalized tracks (not matched in any frame for 1.5 seconds)
        finalized_ids = []
        for tid, meta in track_metadata.items():
            if curr_time - meta['last_seen'] > 1.5:
                finalized_ids.append(tid)
                
        for tid in finalized_ids:
            meta = track_metadata[tid]
            first_seen = meta['first_seen']
            last_seen = meta['last_seen']
            total_tracked_time = last_seen - first_seen
            class_name_guess = meta['class_counts'].most_common(1)[0][0]
            
            buf = track_buffers.get(tid)
            crops = buf.crops if buf else []
            num_crops = len(crops)
            
            if total_tracked_time >= 1.0 and num_crops >= 2:
                print(f"[TRACK LIFECYCLE] Track {tid} ({class_name_guess}) finalized: ACCEPTED. "
                      f"Tracked time: {total_tracked_time:.2f}s, Crops: {num_crops}")
                finalized_queue.put({
                    'track_id': tid,
                    'class_name_guess': class_name_guess,
                    'crops': crops,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                })
            else:
                reason_parts = []
                if total_tracked_time < 1.0:
                    reason_parts.append(f"tracked time {total_tracked_time:.2f}s < 1.0s")
                if num_crops < 2:
                    reason_parts.append(f"crops count {num_crops} < 2")
                reason_str = " & ".join(reason_parts)
                print(f"[TRACK LIFECYCLE] Track {tid} ({class_name_guess}) finalized: DISCARDED as noise ({reason_str}). "
                      f"Tracked time: {total_tracked_time:.2f}s, Crops: {num_crops}")
            
            # Clean up
            if tid in track_metadata:
                del track_metadata[tid]
            if tid in track_buffers:
                del track_buffers[tid]

        # Calculate Running FPS (averaged over the last 30 frames)
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
        cv2.imshow("YOLOE Live Detection Overlay", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalize any remaining tracks at exit
    if track_metadata:
        print(f"\n[TRACK LIFECYCLE] Finalizing remaining {len(track_metadata)} tracks on exit...")
        for tid in list(track_metadata.keys()):
            meta = track_metadata[tid]
            first_seen = meta['first_seen']
            last_seen = meta['last_seen']
            total_tracked_time = last_seen - first_seen
            class_name_guess = meta['class_counts'].most_common(1)[0][0]
            
            buf = track_buffers.get(tid)
            crops = buf.crops if buf else []
            num_crops = len(crops)
            
            if total_tracked_time >= 1.0 and num_crops >= 2:
                print(f"[TRACK LIFECYCLE] Track {tid} ({class_name_guess}) finalized (exit): ACCEPTED. "
                      f"Tracked time: {total_tracked_time:.2f}s, Crops: {num_crops}")
                finalized_queue.put({
                    'track_id': tid,
                    'class_name_guess': class_name_guess,
                    'crops': crops,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                })
            else:
                reason_parts = []
                if total_tracked_time < 1.0:
                    reason_parts.append(f"tracked time {total_tracked_time:.2f}s < 1.0s")
                if num_crops < 2:
                    reason_parts.append(f"crops count {num_crops} < 2")
                reason_str = " & ".join(reason_parts)
                print(f"[TRACK LIFECYCLE] Track {tid} ({class_name_guess}) finalized (exit): DISCARDED as noise ({reason_str}). "
                      f"Tracked time: {total_tracked_time:.2f}s, Crops: {num_crops}")

    cap.release()
    cv2.destroyAllWindows()
    print("Inference loop finished. Webcam released.")

    # Signal the background worker thread to stop and wait for it to process remaining items
    print("\nWaiting for background tagging worker to complete remaining tasks...")
    finalized_queue.put(None)
    worker_thread.join()
    print("Background worker shut down successfully.")

    # Print database summary of processed objects
    try:
        with db.get_db_connection() as conn:
            rows = conn.execute("SELECT track_id, status, tags, ocr_text, confidence FROM objects ORDER BY track_id ASC").fetchall()
        if rows:
            print(f"\n--- SQLite Database Objects (Total: {len(rows)}) ---")
            for row in rows:
                tags_str = row['tags']
                print(f"  - Track ID: {row['track_id']} | Status: {row['status']} | Confidence: {row['confidence']:.2f} | Tags: {tags_str} | OCR: '{row['ocr_text']}'")
    except Exception as db_err:
        print(f"Error querying final database state: {db_err}")

if __name__ == "__main__":
    main()
