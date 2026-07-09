import os
import sys
import time
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../recall_test.db"))
    captures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../recall_captures"))
    
    os.environ["BATMAN_DB_PATH"] = db_path
    os.environ["BATMAN_CAPTURES_DIR"] = captures_dir
    
    # Import the db module only after the environment variables are set
    import db
    
    # Remove existing files if any
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(captures_dir):
        import shutil
        shutil.rmtree(captures_dir)
        
    db.init_db()
    
    # Insert some objects
    fake_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Object 1: Mug with blue logo
    db.insert_pending_object(
        track_id=1,
        first_seen=time.time() - 3600,
        last_seen=time.time() - 3500,
        confidence=0.92,
        crops=[fake_crop]
    )
    db.update_object_result(
        track_id=1,
        tags=["coffee mug", "blue logo", "ceramic cup"],
        ocr_text="Google Cloud",
        status="tagged"
    )
    
    # Object 2: Red backpack
    db.insert_pending_object(
        track_id=2,
        first_seen=time.time() - 1800,
        last_seen=time.time() - 1700,
        confidence=0.88,
        crops=[fake_crop]
    )
    db.update_object_result(
        track_id=2,
        tags=["backpack", "red", "bag"],
        ocr_text="North Face",
        status="tagged"
    )
    
    # Object 3: Blue water bottle
    db.insert_pending_object(
        track_id=3,
        first_seen=time.time() - 600,
        last_seen=time.time() - 500,
        confidence=0.95,
        crops=[fake_crop]
    )
    db.update_object_result(
        track_id=3,
        tags=["water bottle", "blue", "metal flask"],
        ocr_text="Hydro Flask",
        status="tagged"
    )
    
    print(f"Recall test database seeded successfully at {db_path}!")

if __name__ == "__main__":
    main()
