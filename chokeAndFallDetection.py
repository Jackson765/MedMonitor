import cv2
from ultralytics import YOLO
import torch

device = '0' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on: {'NVIDIA GPU' if device == '0' else 'CPU'}")

try:
    print("🧠 Loading Choking Model...")
    choking_model = YOLO('best.pt')  # Your custom model
    
    print("🦴 Loading Fall Model...")
    pose_model = YOLO('yolov8n-pose.pt') # Official model
except Exception as e:
    print(f"❌ Error loading models. Make sure 'best.pt' is in this folder!\n{e}")
    exit()


def genDetection(frame):
    returnNum = 0
    output_frame = frame.copy()

    # --- DETECTION 1: CHOKING ---
    # conf=0.75 keeps it strictly looking for the hand gesture, not just a face
    choking_results = choking_model.predict(source=frame, conf=0.75, device=device, verbose=False)
    
    if len(choking_results[0].boxes) > 0:
        print("choking")
        output_frame = choking_results[0].plot()
        returnNum = 1

    # --- DETECTION 2: FALLING (Stricter Aspect Ratio Logic) ---
    # conf=0.6 to ignore background noise
    pose_results = pose_model.predict(source=frame, conf=0.6, device=device, verbose=False)
    
    if len(pose_results[0].boxes) > 0:
        for box in pose_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            
            # The Logitech Fix: Width must be 30% larger than height (1.3 ratio)
            aspect_ratio = w / h
            
            if aspect_ratio > 1.3:
                # cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                # cv2.putText(output_frame, "!!! FALL DETECTED !!!", (int(x1), int(y1)-15), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                returnNum = 3
    return returnNum