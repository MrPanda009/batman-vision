import os
import sys
import urllib.request

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
RELEASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

MODELS = [
    "yoloe-11s-seg-pf.pt",
    "yoloe-26s-seg-pf.pt",
    "yoloe-26s-seg.pt",
    "mobileclip_blt.ts",
    "mobileclip2_b.ts",
]

def download_file(url, dest_path):
    """Download a file with a progress bar using only the standard library."""
    temp_dest = dest_path + ".tmp"
    try:
        print(f"Downloading {url}...")
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded = 0
            
            with open(temp_dest, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if total_size > 0:
                        percent = min(100, int(downloaded * 100 / total_size))
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        bar_length = 40
                        filled_length = int(bar_length * percent // 100)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        sys.stdout.write(f"\r|{bar}| {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB")
                        sys.stdout.flush()
            print() # New line after progress bar finishes
        
        # Rename temp file to final destination
        os.replace(temp_dest, dest_path)
        print(f"SUCCESS: Saved to {dest_path}\n")
    except Exception as e:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        print(f"\nERROR: Failed to download {url}: {e}\n")
        raise e

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    print("=========================================")
    print("  Batman-Vision Models Downloader        ")
    print("=========================================\n")
    print(f"Models directory: {MODELS_DIR}\n")
    
    for model_name in MODELS:
        dest_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(dest_path):
            print(f"[Exists] Skipping {model_name} (already in models/)")
            continue
        
        url = f"{RELEASE_URL}/{model_name}"
        try:
            download_file(url, dest_path)
        except Exception:
            print("Stopping download process due to error.")
            sys.exit(1)
            
    print("All models verified / downloaded successfully!")

if __name__ == "__main__":
    main()
