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

def update_object_result(track_id, tags, ocr_text, status):
    """Updates the tags, OCR text, and status of an object after processing/tagging.
    
    Args:
        track_id (int): Unique track identifier.
        tags (list, str or None): Tags to associate (JSON list or python list/tuple).
        ocr_text (str or None): Text extracted via OCR.
        status (str): New status ('pending', 'tagged', 'failed').
    """
    if status not in ('pending', 'tagged', 'failed'):
        raise ValueError("status must be one of 'pending', 'tagged', 'failed'")
        
    # Serialize tags if passed as list or tuple
    if tags is not None and not isinstance(tags, str):
        tags_str = json.dumps(tags)
    else:
        tags_str = tags
        
    with get_db_connection() as conn:
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
