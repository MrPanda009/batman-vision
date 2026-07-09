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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load environment variables from .env file
load_dotenv()

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# Initialize SQLite database
print("Initializing SQLite database...")
db.init_db()

app = FastAPI(title="Batman Vision Backend Orchestrator")

# Enable CORS for Next.js app (http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount captures directory to serve crop JPEGs
os.makedirs(db.CAPTURES_DIR, exist_ok=True)
app.mount("/captures", StaticFiles(directory=db.CAPTURES_DIR), name="captures")

# Global pipeline state
frame_lock = threading.Lock()
latest_frame_jpeg = None

capture_thread = None
stop_event = threading.Event()
pipeline_active = False

# Queue for finalized accepted tracks
finalized_queue = queue.Queue()

def is_pipeline_active():
    """Helper to check if the capture thread is alive."""
    global capture_thread
    return capture_thread is not None and capture_thread.is_alive()

# Threshold constants for crop filtering
MIN_SHARPNESS = 50.0   # Minimum Laplacian variance for a crop to be considered sharp
MIN_BBOX_SIZE = 1600   # Minimum bounding box area in pixels (width * height)
MIN_CONFIDENCE = 0.35  # Minimum detection confidence score
DEDUP_SIMILARITY_THRESHOLD = 0.9  # Threshold above which an object is considered a re-sighting

# Weights for the combined quality score (weighted combination of confidence, size, and sharpness)
WEIGHT_CONF = 1.0
WEIGHT_SIZE = 1.0
WEIGHT_SHARP = 1.0

def cv2_to_base64_data_url(img):
    """Encode an OpenCV image (numpy array) to a base64 Data URL."""
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def execute_tagging_with_retries(client, messages):
    """Executes tagging. First tries nvidia/llama-3.1-nemotron-nano-vl-8b-v1 as the primary VLM.
    Falls back to qwen/qwen3.5-397b-a17b.
    As a third option, falls back to nvidia/nemotron-3-nano-omni-30b-a3b-reasoning.
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
            timeout=15.0  # Fail fast if API hangs
        )
        content = response.choices[0].message.content
        result_data = parse_response(content)
        
        confidence = result_data.get("confidence", "low").lower()
        if confidence == "low":
            raise ValueError("Fallback VLM 1 (Qwen) returned low confidence")
            
        print("[WORKER] Fallback VLM 1 (Qwen) tagging successful.")
        return result_data
        
    except Exception as err:
        print(f"[WORKER] Fallback VLM 1 (qwen/qwen3.5-397b-a17b) failed or timed out: {err}. Trying Nemotron Reasoning...")

    # 3. Fallback VLM 2: Nemotron 3 Nano Omni (with reasoning)
    print("[WORKER] Invoking fallback VLM 2 (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)...")
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":4096},
            timeout=15.0  # Fail fast if API hangs
        )
        content = response.choices[0].message.content
        result_data = parse_response(content)
        
        print("[WORKER] Fallback VLM 2 (Nemotron Reasoning) tagging successful.")
        return result_data
        
    except Exception as err:
        print(f"[WORKER] All VLMs failed or timed out. Final error: {err}")
        raise err

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
            result = execute_tagging_with_retries(client, messages)
            
            # Combine object name into tags list so it displays prominently
            tags_list = result['tags']
            obj_name = result['object_name']
            if obj_name and obj_name.lower() not in [t.lower() for t in tags_list]:
                tags_list.insert(0, obj_name)

            # Write back result (pending -> tagged), and store embedding once tagging succeeds
            db.update_object_result(
                track_id=track_id,
                tags=tags_list,
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

def compute_blur_score(crop):
    """Compute the Laplacian variance as a sharpness/blur score."""
    if crop is None or crop.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes."""
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
        self.crops = []

    def add_crop(self, crop_img, bbox, confidence, sharpness, score):
        new_crop = {
            'image': crop_img.copy(),
            'bbox': list(bbox),
            'confidence': confidence,
            'sharpness': sharpness,
            'score': score
        }
        
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
        
        if len(self.crops) < self.max_crops:
            self.crops.append(new_crop)
            self.crops.sort(key=lambda x: x['score'], reverse=True)
            return True, f"added (capacity: {len(self.crops)}/{self.max_crops})"
        
        worst_crop = self.crops[-1]
        if score > worst_crop['score']:
            self.crops[-1] = new_crop
            self.crops.sort(key=lambda x: x['score'], reverse=True)
            return True, f"evicted worst (score: {worst_crop['score']:.2f}) and replaced with new (score: {score:.2f})"
            
        return False, f"rejected (score {score:.2f} <= worst score {worst_crop['score']:.2f})"

def capture_loop_func():
    """Background thread function that runs the webcam frame capture,
    running tracking via YOLOE, filtering crop images, updating TrackBuffers,
    and finalising objects.
    """
    global latest_frame_jpeg, pipeline_active
    
    print("[PIPELINE] Capture thread starting...")
    mps_available = torch.backends.mps.is_available()
    device = "mps" if mps_available else "cpu"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models/yoloe-26l-seg-pf.pt")
    tracker_path = os.path.join(script_dir, "tests/custom_bytetrack.yaml")
    
    if not os.path.exists(model_path):
        print(f"[PIPELINE ERROR] Model file not found at {model_path}!")
        pipeline_active = False
        return
        
    try:
        model = YOLOE(model_path)
    except Exception as e:
        print(f"[PIPELINE ERROR] Failed to load YOLOE model: {e}")
        pipeline_active = False
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[PIPELINE ERROR] Could not open webcam.")
        pipeline_active = False
        return

    print("[PIPELINE] Webcam successfully opened. Starting capture loop.")
    pipeline_active = True
    
    frame_times = collections.deque(maxlen=30)
    prev_time = time.perf_counter()
    hud_color = (255, 255, 0)
    active_tracks = {}
    track_buffers = {}
    track_metadata = {}
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[PIPELINE] Failed to read frame from webcam.")
                time.sleep(0.03)
                continue

            frame_clean = frame.copy()
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            frame_times.append(dt)
            inst_fps = 1.0 / dt if dt > 0 else 0.0

            results = model.track(
                frame, 
                persist=True, 
                tracker=tracker_path, 
                device=device, 
                conf=0.1, 
                verbose=False
            )
            
            current_tracks = {}

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else []
                
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names.get(cls_id, f"class_{cls_id}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), hud_color, 1)
                    
                    if has_ids and i < len(track_ids):
                        track_id = track_ids[i]
                        current_tracks[track_id] = class_name
                        label_text = f"ID {track_id} | {class_name} {conf:.2f}"
                        
                        if track_id not in track_metadata:
                            track_metadata[track_id] = {
                                'first_seen': curr_time,
                                'last_seen': curr_time,
                                'class_counts': collections.Counter({class_name: 1})
                            }
                        else:
                            track_metadata[track_id]['last_seen'] = curr_time
                            track_metadata[track_id]['class_counts'][class_name] += 1
                        
                        if track_id not in track_buffers:
                            track_buffers[track_id] = TrackBuffer()
                        
                        h, w = frame_clean.shape[:2]
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(w, x2), min(h, y2)
                        
                        if x2_c > x1_c and y2_c > y1_c:
                            crop_img = frame_clean[y1_c:y2_c, x1_c:x2_c]
                            bbox_size = (x2_c - x1_c) * (y2_c - y1_c)
                            
                            if conf >= MIN_CONFIDENCE and bbox_size >= MIN_BBOX_SIZE:
                                sharpness = compute_blur_score(crop_img)
                                if sharpness >= MIN_SHARPNESS:
                                    score = compute_combined_score(conf, bbox_size, sharpness)
                                    track_buffers[track_id].add_crop(
                                        crop_img, [x1_c, y1_c, x2_c, y2_c], conf, sharpness, score
                                    )
                    else:
                        label_text = f"{class_name} {conf:.2f}"
                    
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    tx = x1
                    ty = y1 - 4
                    if ty - text_h < 0:
                        ty = y1 + text_h + 4
                    
                    bg_pt1 = (tx, ty - text_h - 2)
                    bg_pt2 = (tx + text_w + 4, ty + baseline)
                    draw_semi_transparent_rect(frame, bg_pt1, bg_pt2, (0, 0, 0), 0.6)
                    
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

            new_ids = set(current_tracks.keys()) - set(active_tracks.keys())
            disappeared_ids = set(active_tracks.keys()) - set(current_tracks.keys())
            
            for tid in new_ids:
                print(f"[TRACKER] [FPS: {inst_fps:.1f}] New track ID appeared: ID {tid} ({current_tracks[tid]})")
            for tid in disappeared_ids:
                print(f"[TRACKER] [FPS: {inst_fps:.1f}] Track ID disappeared: ID {tid} ({active_tracks[tid]})")
                
            active_tracks = current_tracks

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
                
                if tid in track_metadata:
                    del track_metadata[tid]
                if tid in track_buffers:
                    del track_buffers[tid]

            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            fps_text = f"FPS: {fps:.1f}"
            
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

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            with frame_lock:
                latest_frame_jpeg = frame_bytes

    except Exception as e:
        print(f"[PIPELINE ERROR] Exception in capture thread: {e}")
    finally:
        # Finalize any remaining tracks
        if track_metadata:
            print(f"[TRACK LIFECYCLE] Finalizing remaining {len(track_metadata)} tracks on exit...")
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
                    print(f"[TRACK LIFECYCLE] Track {tid} ({class_name_guess}) finalized (exit): ACCEPTED.")
                    finalized_queue.put({
                        'track_id': tid,
                        'class_name_guess': class_name_guess,
                        'crops': crops,
                        'first_seen': first_seen,
                        'last_seen': last_seen
                    })
        
        cap.release()
        print("[PIPELINE] Webcam released.")
        with frame_lock:
            latest_frame_jpeg = None
        pipeline_active = False

# Start continuous background worker
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models/yoloe-26l-seg-pf.pt")

worker_thread = threading.Thread(
    target=tagging_worker_func,
    args=(finalized_queue, model_path),
    daemon=True
)
worker_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    global capture_thread
    print("[SERVER] Shutting down backend...")
    # Stop capture thread if running
    stop_event.set()
    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=5.0)
        
    # Stop background worker
    finalized_queue.put(None)
    worker_thread.join(timeout=5.0)
    print("[SERVER] Clean shutdown complete.")

# ----------------- HTTP API ENDPOINTS -----------------

def generate_video_stream():
    """Generates MJPEG stream of latest HUD-annotated frames."""
    # Base offline frame
    offline_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(offline_frame, "Pipeline Stopped", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    _, offline_buf = cv2.imencode('.jpg', offline_frame)
    offline_bytes = offline_buf.tobytes()

    while True:
        is_active = is_pipeline_active()
        if is_active:
            with frame_lock:
                frame_bytes = latest_frame_jpeg
            
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # limit stream to ~30 FPS
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + offline_bytes + b'\r\n')
            time.sleep(1.0)

@app.get("/video_feed")
def video_feed():
    """Returns multipart stream of HUD live view."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/pipeline/status")
def get_pipeline_status():
    """Returns whether the webcam capture pipeline is currently active."""
    return {"active": is_pipeline_active()}

@app.post("/pipeline/start")
def start_pipeline():
    """Starts background camera capture and object tracking thread."""
    global capture_thread, stop_event, pipeline_active
    
    if is_pipeline_active():
        return {"status": "already_running", "message": "Pipeline is already active."}
        
    stop_event.clear()
    pipeline_active = False
    
    capture_thread = threading.Thread(target=capture_loop_func, daemon=True)
    capture_thread.start()
    
    # Wait up to 3 seconds for the webcam to initialize
    start_wait = time.time()
    while time.time() - start_wait < 3.0:
        if pipeline_active:
            return {"status": "started", "message": "Pipeline started successfully."}
        time.sleep(0.1)
        
    if capture_thread.is_alive():
        return {"status": "starting", "message": "Pipeline is initializing."}
    else:
        raise HTTPException(status_code=500, detail="Failed to initialize webcam pipeline.")

@app.post("/pipeline/stop")
def stop_pipeline():
    """Stops background camera capture and cleanly releases webcam."""
    global capture_thread, stop_event
    
    if not is_pipeline_active():
        return {"status": "already_stopped", "message": "Pipeline is not running."}
        
    stop_event.set()
    if capture_thread is not None:
        capture_thread.join(timeout=5.0)
        
    return {"status": "stopped", "message": "Pipeline stopped successfully and camera released."}

@app.get("/api/objects")
def get_objects():
    """Returns JSON for all rows in objects table, most recent first."""
    try:
        with db.get_db_connection() as conn:
            rows = conn.execute(
                "SELECT track_id, tags, ocr_text, status, first_seen, last_seen, crop_paths FROM objects ORDER BY last_seen DESC"
            ).fetchall()
            
        results = []
        for r in rows:
            try:
                crop_paths = json.loads(r["crop_paths"]) if r["crop_paths"] else []
            except Exception:
                crop_paths = []
                
            try:
                tags = json.loads(r["tags"]) if r["tags"] else []
            except Exception:
                tags = []
                
            results.append({
                "id": r["track_id"],
                "tags": tags,
                "ocr_text": r["ocr_text"] or "",
                "status": r["status"],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "crop_paths": crop_paths
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

@app.get("/api/stats")
def get_stats():
    """Returns counts of pending, tagged, and failed objects."""
    try:
        with db.get_db_connection() as conn:
            rows = conn.execute(
                "SELECT status, count(*) as count FROM objects GROUP BY status"
            ).fetchall()
            
        stats = {"pending": 0, "tagged": 0, "failed": 0}
        for r in rows:
            status = r["status"]
            if status in stats:
                stats[status] = r["count"]
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

@app.post("/api/clear")
def clear_database():
    """Clears all objects from database and deletes crop files.
    Fails with 400 Bad Request if the camera pipeline is active.
    """
    if is_pipeline_active():
        raise HTTPException(
            status_code=400,
            detail="Cannot clear database while the webcam pipeline is active."
        )
        
    try:
        with db.get_db_connection() as conn:
            count_row = conn.execute("SELECT count(*) as count FROM objects").fetchone()
            count = count_row["count"] if count_row else 0
            
            # Wiping database rows. Triggers automatically clean up FTS5.
            conn.execute("DELETE FROM objects")
            conn.commit()
            
        # Delete crop files in CAPTURES_DIR
        cleared_files = 0
        if os.path.exists(db.CAPTURES_DIR):
            for filename in os.listdir(db.CAPTURES_DIR):
                file_path = os.path.join(db.CAPTURES_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        cleared_files += 1
                except Exception as e:
                    print(f"[CLEAR] Error deleting file {file_path}: {e}")
                    
        return {
            "cleared_objects": count,
            "deleted_files": cleared_files,
            "message": "Database and crop files cleared successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear operation failed: {e}")
