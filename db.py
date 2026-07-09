import os
import json
import sqlite3
import time
import cv2

# Default database path (in the root directory of the workspace)
DB_PATH = os.environ.get(
    "BATMAN_DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "objects.db"))
)

# Default captures directory (in the root directory of the workspace)
CAPTURES_DIR = os.environ.get(
    "BATMAN_CAPTURES_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "captures"))
)

def get_db_connection():
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the objects table, FTS5 virtual table, and triggers."""
    with get_db_connection() as conn:
        # Create objects table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                track_id INTEGER PRIMARY KEY,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                crop_paths TEXT NOT NULL,  -- JSON array of file paths
                tags TEXT,                 -- JSON array of tags, nullable
                ocr_text TEXT,             -- OCR result text, nullable
                status TEXT NOT NULL CHECK(status IN ('pending', 'tagged', 'failed')),
                confidence REAL NOT NULL,
                embedding BLOB,            -- Serialized float32 bytes, nullable
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create FTS5 virtual table for keyword search
        # FTS5 tables automatically index text fields. track_id is marked as UNINDEXED.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(
                track_id UNINDEXED,
                tags,
                ocr_text
            );
        """)
        
        # Create trigger: after insert
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS objects_after_insert
            AFTER INSERT ON objects
            BEGIN
                INSERT INTO objects_fts(track_id, tags, ocr_text)
                VALUES (new.track_id, new.tags, new.ocr_text);
            END;
        """)
        
        # Create trigger: after update (only on tags or ocr_text change)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS objects_after_update
            AFTER UPDATE OF tags, ocr_text ON objects
            BEGIN
                UPDATE objects_fts
                SET tags = new.tags,
                    ocr_text = new.ocr_text
                WHERE track_id = new.track_id;
            END;
        """)
        
        # Create trigger: after delete
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS objects_after_delete
            AFTER DELETE ON objects
            BEGIN
                DELETE FROM objects_fts WHERE track_id = old.track_id;
            END;
        """)
        conn.commit()

def save_crops_to_disk(track_id, crops, captures_dir=None):
    """Saves crop images to disk and returns a list of their relative paths.
    
    Args:
        track_id (int): Unique track identifier (used in filenames).
        crops (list): List of crop images (either numpy arrays, or dictionaries with 'image' key).
        captures_dir (str, optional): Custom path to captures directory.
        
    Returns:
        list: List of relative paths of saved crop files.
    """
    if captures_dir is None:
        captures_dir = CAPTURES_DIR
        
    os.makedirs(captures_dir, exist_ok=True)
    
    crop_paths = []
    timestamp = int(time.time())
    
    for idx, crop in enumerate(crops):
        # Handle crop if it is a dictionary (like in the TrackBuffer list) or a raw numpy array
        if isinstance(crop, dict) and "image" in crop:
            img = crop["image"]
        else:
            img = crop
            
        filename = f"track_{track_id}_{idx}_{timestamp}.jpg"
        abs_path = os.path.join(captures_dir, filename)
        
        # Save crop image using opencv
        cv2.imwrite(abs_path, img)
        
        # Store relative path for portability
        project_root = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.relpath(abs_path, project_root)
        crop_paths.append(rel_path)
        
    return crop_paths

def insert_pending_object(track_id, first_seen, last_seen, confidence, crops, captures_dir=None, embedding=None):
    """Saves crops to disk under the captures directory and inserts a 'pending' object row.
    
    Args:
        track_id (int): Unique track identifier.
        first_seen (float): Unix epoch timestamp of first detection.
        last_seen (float): Unix epoch timestamp of last detection.
        confidence (float): Bounding box / tracker confidence score.
        crops (list): List of crop images (either numpy arrays, or dictionaries with 'image' key).
        captures_dir (str, optional): Custom path to captures directory.
        embedding (bytes, optional): Serialized embedding bytes (BLOB) for the object.
        
    Returns:
        str: JSON string of relative paths of saved crop files.
    """
    crop_paths = save_crops_to_disk(track_id, crops, captures_dir)
    crop_paths_json = json.dumps(crop_paths)
    
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO objects (track_id, first_seen, last_seen, crop_paths, status, confidence, embedding)
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            (track_id, first_seen, last_seen, crop_paths_json, confidence, embedding)
        )
        conn.commit()
        
    return crop_paths_json

def update_object_re_sighting(matched_track_id, last_seen, new_crops, captures_dir=None):
    """Saves new crops to disk, appends their paths to the existing object's crop_paths, and updates last_seen.
    
    Args:
        matched_track_id (int): Unique track identifier of the existing matched object.
        last_seen (float): New last_seen timestamp.
        new_crops (list): List of new crop images.
        captures_dir (str, optional): Custom path to captures directory.
        
    Returns:
        str: JSON string of combined relative paths.
    """
    new_crop_paths = save_crops_to_disk(matched_track_id, new_crops, captures_dir)
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT crop_paths FROM objects WHERE track_id = ?", (matched_track_id,)).fetchone()
        if row and row['crop_paths']:
            try:
                existing_paths = json.loads(row['crop_paths'])
            except Exception:
                existing_paths = []
        else:
            existing_paths = []
            
        combined_paths = existing_paths + new_crop_paths
        combined_paths_json = json.dumps(combined_paths)
        
        conn.execute(
            """
            UPDATE objects
            SET last_seen = ?,
                crop_paths = ?
            WHERE track_id = ?
            """,
            (last_seen, combined_paths_json, matched_track_id)
        )
        conn.commit()
        
    return combined_paths_json

def update_object_result(track_id, tags, ocr_text, status, embedding=None):
    """Updates the tags, OCR text, status, and optionally the embedding of an object.
    
    Args:
        track_id (int): Unique track identifier.
        tags (list, str or None): Tags to associate (JSON list or python list/tuple).
        ocr_text (str or None): Text extracted via OCR.
        status (str): New status ('pending', 'tagged', 'failed').
        embedding (bytes, optional): Serialized embedding bytes (BLOB) for the object.
    """
    if status not in ('pending', 'tagged', 'failed'):
        raise ValueError("status must be one of 'pending', 'tagged', 'failed'")
        
    # Serialize tags if passed as list or tuple
    if tags is not None and not isinstance(tags, str):
        tags_str = json.dumps(tags)
    else:
        tags_str = tags
        
    with get_db_connection() as conn:
        if embedding is not None:
            conn.execute(
                """
                UPDATE objects
                SET tags = ?,
                    ocr_text = ?,
                    status = ?,
                    embedding = ?
                WHERE track_id = ?
                """,
                (tags_str, ocr_text, status, embedding, track_id)
            )
        else:
            conn.execute(
                """
                UPDATE objects
                SET tags = ?,
                    ocr_text = ?,
                    status = ?
                WHERE track_id = ?
                """,
                (tags_str, ocr_text, status, track_id)
            )
        conn.commit()

def search_objects(query_text):
    """Performs an FTS5 search on objects_fts and returns matching objects.
    
    Args:
        query_text (str): Search term / query string (using FTS5 match grammar).
        
    Returns:
        list: List of matching rows converted to dictionaries with decoded JSON.
    """
    if not query_text or not query_text.strip():
        return []
        
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT o.*
            FROM objects o
            JOIN objects_fts fts ON o.track_id = fts.track_id
            WHERE objects_fts MATCH ?
            """,
            (query_text,)
        )
        rows = cursor.fetchall()
        
    results = []
    for row in rows:
        row_dict = dict(row)
        # Decode JSON arrays
        if row_dict.get("crop_paths"):
            try:
                row_dict["crop_paths"] = json.loads(row_dict["crop_paths"])
            except Exception:
                pass
        if row_dict.get("tags"):
            try:
                row_dict["tags"] = json.loads(row_dict["tags"])
            except Exception:
                pass
        results.append(row_dict)
        
    return results
