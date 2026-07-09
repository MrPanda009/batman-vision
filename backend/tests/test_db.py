import os
import sys
import unittest
import numpy as np
import shutil
import time

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variables to isolate test files
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_objects.db"))
TEST_CAPTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_captures"))

os.environ["BATMAN_DB_PATH"] = TEST_DB_PATH
os.environ["BATMAN_CAPTURES_DIR"] = TEST_CAPTURES_DIR

# Import the db module under test
import db

class TestDatabaseModule(unittest.TestCase):
    
    def setUp(self):
        # Ensure clean state for each test
        self.tearDown()
        db.init_db()

    def tearDown(self):
        # Remove database file if exists
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception as e:
                print(f"Error removing test DB: {e}")
                
        # Remove captures directory if exists
        if os.path.exists(TEST_CAPTURES_DIR):
            try:
                shutil.rmtree(TEST_CAPTURES_DIR)
            except Exception as e:
                print(f"Error removing test captures dir: {e}")

    def test_database_flow(self):
        # 1. Generate fake crops (numpy arrays representing RGB images)
        fake_crop_1 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a colored rectangle to make it look like a real crop
        fake_crop_1[10:90, 10:90] = [0, 255, 0] # Green box
        
        fake_crop_2 = np.zeros((150, 120, 3), dtype=np.uint8)
        fake_crop_2[20:130, 20:100] = [0, 0, 255] # Red box
        
        # 2. Insert two pending objects
        first_seen = time.time() - 5.0
        last_seen = time.time()
        
        print("Inserting fake pending objects...")
        db.insert_pending_object(
            track_id=101,
            first_seen=first_seen,
            last_seen=last_seen,
            confidence=0.88,
            crops=[fake_crop_1]
        )
        
        db.insert_pending_object(
            track_id=102,
            first_seen=first_seen - 2.0,
            last_seen=last_seen - 1.0,
            confidence=0.75,
            crops=[fake_crop_2]
        )
        
        # Check files were created in the test captures directory
        captures_files = os.listdir(TEST_CAPTURES_DIR)
        self.assertEqual(len(captures_files), 2)
        print(f"Verified crops saved to disk: {captures_files}")
        
        # Check database rows in objects
        with db.get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM objects ORDER BY track_id ASC").fetchall()
            self.assertEqual(len(rows), 2)
            
            # Row 1 check
            self.assertEqual(rows[0]['track_id'], 101)
            self.assertEqual(rows[0]['status'], 'pending')
            self.assertEqual(rows[0]['confidence'], 0.88)
            self.assertIsNone(rows[0]['tags'])
            
            # Row 2 check
            self.assertEqual(rows[1]['track_id'], 102)
            self.assertEqual(rows[1]['status'], 'pending')
            self.assertEqual(rows[1]['confidence'], 0.75)
            self.assertIsNone(rows[1]['tags'])
            
        print("Database pending insertion verified successfully.")
        
        # 3. Update objects with tag results
        print("Updating objects with tagging results...")
        db.update_object_result(
            track_id=101,
            tags=["person", "backpack", "jacket"],
            ocr_text="NYU Athletics",
            status="tagged"
        )
        
        db.update_object_result(
            track_id=102,
            tags=["bicycle", "helmet"],
            ocr_text="TREK 7200",
            status="tagged"
        )
        
        # Check database rows after update
        with db.get_db_connection() as conn:
            rows = conn.execute("SELECT * FROM objects ORDER BY track_id ASC").fetchall()
            self.assertEqual(rows[0]['status'], 'tagged')
            self.assertEqual(rows[1]['status'], 'tagged')
            self.assertEqual(rows[0]['ocr_text'], 'NYU Athletics')
            self.assertEqual(rows[1]['ocr_text'], 'TREK 7200')
            
        # 4. Perform FTS5 searches and confirm matching rows
        print("Verifying search_objects functionality...")
        
        # Search by tag
        res_backpack = db.search_objects("backpack")
        self.assertEqual(len(res_backpack), 1)
        self.assertEqual(res_backpack[0]['track_id'], 101)
        self.assertEqual(res_backpack[0]['tags'], ["person", "backpack", "jacket"])
        print("  - Search for 'backpack' found track 101 successfully.")
        
        res_bicycle = db.search_objects("bicycle")
        self.assertEqual(len(res_bicycle), 1)
        self.assertEqual(res_bicycle[0]['track_id'], 102)
        print("  - Search for 'bicycle' found track 102 successfully.")
        
        # Search by OCR text
        res_trek = db.search_objects("TREK")
        self.assertEqual(len(res_trek), 1)
        self.assertEqual(res_trek[0]['track_id'], 102)
        print("  - Search for 'TREK' found track 102 successfully.")
        
        # Search by OCR text part
        res_athletics = db.search_objects("Athletics")
        self.assertEqual(len(res_athletics), 1)
        self.assertEqual(res_athletics[0]['track_id'], 101)
        print("  - Search for 'Athletics' found track 101 successfully.")
        
        # Search multiple terms
        res_multi = db.search_objects("person OR helmet")
        self.assertEqual(len(res_multi), 2)
        print("  - Search for 'person OR helmet' returned both tracks successfully.")
        
        # Search nonexistent
        res_none = db.search_objects("car")
        self.assertEqual(len(res_none), 0)
        print("  - Search for nonexistent term 'car' returned 0 results as expected.")
        
        # 5. Verify trigger propagation on DELETE
        print("Verifying DELETE propagation...")
        with db.get_db_connection() as conn:
            conn.execute("DELETE FROM objects WHERE track_id = 101")
            conn.commit()
            
        # Search again for 'backpack' should return 0 results
        res_after_delete = db.search_objects("backpack")
        self.assertEqual(len(res_after_delete), 0)
        print("  - Search for 'backpack' returned 0 results after deletion successfully.")
        
        # Search for 'bicycle' should still work
        res_still_there = db.search_objects("bicycle")
        self.assertEqual(len(res_still_there), 1)
        print("  - Search for 'bicycle' still active and returned 1 result successfully.")

if __name__ == "__main__":
    unittest.main()
