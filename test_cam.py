import cv2
import numpy as np

print("Opening camera with different backends...")

# Try different camera indices
for i in range(3):
    print(f"\nTrying camera index {i}...")
    
    # Try with DirectShow backend (usually works better on Windows)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        print(f"Camera {i} opened successfully with DirectShow")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Frame captured successfully! Shape: {frame.shape}")
            cv2.imshow(f'Camera {i} Test', frame)
            print("Press 'q' to quit")
            
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
            cap.release()
            break
        else:
            print(f"Camera {i} opened but couldn't read frame")
            cap.release()
    else:
        print(f"Camera {i} could not be opened")

print("Test complete")