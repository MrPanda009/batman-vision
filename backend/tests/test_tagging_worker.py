import os
import sys
import time
import queue
import threading
import numpy as np
import cv2
import json
from ultralytics import YOLOE

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Isolate database for tests
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_worker_objects.db"))
TEST_CAPTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_worker_captures"))

os.environ["BATMAN_DB_PATH"] = TEST_DB_PATH
os.environ["BATMAN_CAPTURES_DIR"] = TEST_CAPTURES_DIR

import db
from yoloe_test import tagging_worker_func
finalized_queue = queue.Queue()

def generate_test_images():
    """Generates distinct synthetic object crop images."""
    # Object A: Red Cup (Crop 1)
    cup1 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.rectangle(cup1, (30, 30), (120, 130), (0, 0, 255), -1)  # Red box
    cv2.putText(cup1, "COFFEE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Object A: Red Cup (Crop 2 - slightly shifted/variant)
    cup2 = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.rectangle(cup2, (32, 32), (122, 132), (0, 0, 255), -1)  # Red box
    cv2.putText(cup2, "COFFEE", (42, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Object B: Green Bottle (Gray background to differ from Red Cup)
    bottle = np.ones((150, 150, 3), dtype=np.uint8) * 128
    cv2.rectangle(bottle, (50, 20), (100, 140), (0, 255, 0), -1)  # Green box
    cv2.putText(bottle, "WATER", (52, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return cup1, cup2, bottle

def main():
    print("==================================================")
    # 1. Clean test DB
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_CAPTURES_DIR):
        import shutil
        shutil.rmtree(TEST_CAPTURES_DIR)
        
    db.init_db()
    print("Test database initialized.")
    
    # 2. Get YOLOE model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "../models/yoloe-26l-seg-pf.pt"))
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please run download_models.py first.")
        sys.exit(1)
        
    # Check if NVIDIA_API_KEY is available
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY environment variable is missing. Set it in .env or your terminal.")
        sys.exit(1)
    print("NVIDIA_API_KEY detected.")

    # 3. Start background worker thread
    print("Starting background worker thread...")
    worker_thread = threading.Thread(
        target=tagging_worker_func,
        args=(finalized_queue, model_path),
        daemon=True
    )
    worker_thread.start()
    
    # 4. Generate synthetic test crops
    cup1, cup2, bottle = generate_test_images()
    
    # --- TEST CASE 1: Tagging a new object (Cup) ---
    print("\n--- TEST CASE 1: Tagging Cup ---")
    track_1_crops = [
        {'image': cup1, 'confidence': 0.9, 'bbox': [10, 10, 140, 140], 'score': 1.5},
        {'image': cup2, 'confidence': 0.85, 'bbox': [12, 12, 142, 142], 'score': 1.4}
    ]
    
    finalized_queue.put({
        'track_id': 1001,
        'crops': track_1_crops,
        'first_seen': time.time() - 3,
        'last_seen': time.time(),
        'class_name_guess': 'cup'
    })
    
    # Wait for queue to process
    print("Waiting for worker to process Track 1001...")
    finalized_queue.join()
    print("Track 1001 processed.")
    
    # --- TEST CASE 2: De-duplication (Same Cup walked back in) ---
    print("\n--- TEST CASE 2: De-duplication (Cup walked back in) ---")
    # We use cup2 again as crops under a new track ID (1002)
    track_2_crops = [
        {'image': cup2, 'confidence': 0.88, 'bbox': [11, 11, 141, 141], 'score': 1.45},
        {'image': cup1, 'confidence': 0.86, 'bbox': [13, 13, 143, 143], 'score': 1.38}
    ]
    
    finalized_queue.put({
        'track_id': 1002,
        'crops': track_2_crops,
        'first_seen': time.time() - 2,
        'last_seen': time.time(),
        'class_name_guess': 'cup'
    })
    
    # Wait for queue to process
    print("Waiting for worker to process Track 1002 (should dedup)...")
    finalized_queue.join()
    print("Track 1002 processed.")
    
    # --- TEST CASE 3: Different Object (Bottle) ---
    print("\n--- TEST CASE 3: Tagging different object (Bottle) ---")
    track_3_crops = [
        {'image': bottle, 'confidence': 0.92, 'bbox': [20, 20, 130, 130], 'score': 1.6}
    ]
    
    # Note: track lifecycle requirements mandate at least 2 crops, but let's pass 2 crops of the bottle to make it realistic
    bottle_crop_variant = np.ones((150, 150, 3), dtype=np.uint8) * 128
    cv2.rectangle(bottle_crop_variant, (51, 21), (101, 141), (0, 255, 0), -1)
    cv2.putText(bottle_crop_variant, "WATER", (53, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    track_3_crops.append({
        'image': bottle_crop_variant, 'confidence': 0.91, 'bbox': [21, 21, 131, 131], 'score': 1.55
    })
    
    finalized_queue.put({
        'track_id': 1003,
        'crops': track_3_crops,
        'first_seen': time.time() - 4,
        'last_seen': time.time(),
        'class_name_guess': 'bottle'
    })
    
    # Wait for queue to process
    print("Waiting for worker to process Track 1003...")
    finalized_queue.join()
    print("Track 1003 processed.")

    # --- TEST CASE 4: Network Error Handling (Invalid API Endpoint) ---
    print("\n--- TEST CASE 4: Network Error Handling ---")
    # We will temporarily point the client base_url to an invalid address to force a network error
    # Let's stop the running worker thread
    finalized_queue.put(None)
    worker_thread.join()
    
    # Start a new worker thread with an invalid API URL
    print("Starting a new worker thread with invalid API URL to simulate network failure...")
    os.environ["NVIDIA_API_KEY"] = "dummy_key"
    
    # To simulate network failure, we temporarily override the OpenAI base_url inside the worker.
    # We can write a custom worker function or temporarily mock the client.
    # Let's define a test worker target function that points to a dead URL:
    from openai import OpenAI
    
    def failing_tagging_worker_func(finalized_queue, model_path):
        import db
        embedding_model = YOLOE(model_path)
        bad_client = OpenAI(
            base_url="https://invalid-dead-end-integrate.api.nvidia.com/v1",
            api_key="bad_key"
        )
        
        while True:
            item = finalized_queue.get()
            if item is None:
                finalized_queue.task_done()
                break
            
            track_id = item['track_id']
            # We bypass embedding / dedup check to directly test API failure and 'failed' status update
            try:
                # Insert pending row (None -> pending)
                db.insert_pending_object(
                    track_id=track_id,
                    first_seen=item['first_seen'],
                    last_seen=item['last_seen'],
                    confidence=0.9,
                    crops=item['crops']
                )
                print(f"[STATUS] Track {track_id} transition: None -> pending")
                
                # Try calling API (should raise ConnectionError / tenacity retries)
                from yoloe_test import execute_tagging_with_retries
                
                # Mock a small call
                execute_tagging_with_retries(bad_client, [{"role": "user", "content": "test"}])
                
            except Exception as e:
                print(f"[WORKER ERROR] Permanent tagging failure for track {track_id}: {e}")
                db.update_object_result(
                    track_id=track_id,
                    tags=None,
                    ocr_text=None,
                    status='failed'
                )
                print(f"[STATUS] Track {track_id} transition: pending -> failed")
            finally:
                finalized_queue.task_done()

    failing_worker = threading.Thread(
        target=failing_tagging_worker_func,
        args=(finalized_queue, model_path),
        daemon=True
    )
    failing_worker.start()
    
    # Queue up a new track to test failure
    finalized_queue.put({
        'track_id': 1004,
        'crops': track_1_crops, # Reuse cup crops
        'first_seen': time.time() - 2,
        'last_seen': time.time(),
        'class_name_guess': 'cup'
    })
    
    print("Waiting for worker to process Track 1004 (should fail due to bad URL)...")
    finalized_queue.join()
    print("Track 1004 processed.")
    
    # Clean up failing worker
    finalized_queue.put(None)
    failing_worker.join()

    # --- PRINT DATABASE FINAL STATE ---
    print("\n--- Final Database Contents ---")
    with db.get_db_connection() as conn:
        rows = conn.execute("SELECT track_id, status, confidence, tags, ocr_text, crop_paths FROM objects").fetchall()
        for r in rows:
            try:
                paths = json.loads(r['crop_paths'])
                crop_count = len(paths)
            except Exception:
                crop_count = 0
            print(f"Track {r['track_id']} | Status: {r['status']} | Confidence: {r['confidence']:.2f} | Crops: {crop_count} | Tags: {r['tags']} | OCR: '{r['ocr_text']}'")
            
    # Clean up files
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_CAPTURES_DIR):
        import shutil
        shutil.rmtree(TEST_CAPTURES_DIR)
        
    print("\nVerification completed successfully!")

if __name__ == "__main__":
    main()
