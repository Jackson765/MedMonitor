import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box

# 1. Load your face model
model = YOLO('yolov8n-face.pt')

def runFaceDetection(image):
    # 2. Run detection
    results = model(image, conf=0.65, verbose=False)
    savedAnything = False

    # 3. Process detections
    for result in results:
        for i, box in enumerate(result.boxes):
            face_crop = save_one_box(box.xyxy, result.orig_img, save=False)
            savedAnything = True
            filename = 'faceimage.jpg'
            cv2.imwrite(filename, face_crop)
    
    return savedAnything