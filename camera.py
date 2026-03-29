import os
from strokeAI import detect
from comms import inDanger, setDanger, setDangerToZero
from faceDetectionAI import runFaceDetection
import cv2
import threading
import time

dangerThread = threading.Thread(target=inDanger)
# Open the default camera
# Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
def startCam():
    cam = cv2.VideoCapture(0)
    while True:
        time.sleep(.01)
        if cam.isOpened():
            ret, frame = cam.read()

            cv2.imwrite("frame.jpg", frame)

            # Write the frame to the output file
            # out.write(frame)
            # Display the captured frame
            cv2.imshow('Camera', frame)
            if (runFaceDetection("frame.jpg")):
                detectionValue = detect("faceimage.jpg")
                print(detectionValue)
                if (detectionValue >= .9):
                    print("DANGER")
                    setDanger()
                else:
                    setDangerToZero()
            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break
    cam.release()
    cv2.destroyAllWindows()
    os._exit(0)

# Release the capture and writer objects