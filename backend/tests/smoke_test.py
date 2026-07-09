import sys
import cv2
import torch

def main():
    print("Checking torch MPS availability...")
    mps_available = torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_available(): {mps_available}")
    
    if not mps_available:
        print("WARNING: MPS (Metal Performance Shaders) is not available.")
    else:
        print("SUCCESS: MPS is available.")

    print("Opening default webcam (device 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Webcam successfully opened. Press 'q' in the window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        cv2.imshow("Webcam Smoke Test", frame)

        # Wait for 1 ms and check if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and window closed. Smoke test finished.")

if __name__ == "__main__":
    main()
