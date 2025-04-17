
import cv2
from detectors.face_recognition_detector import FaceRecognitionDetector
import torch
import cv2


def get_device():
    """Returns the best available device for PyTorch."""

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device
device = get_device()
print(device)


cap = cv2.VideoCapture(0)  # Open the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FaceRecognition = FaceRecognitionDetector(device)

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Failed to capture frame!")
        break

    newFrame = FaceRecognition.detect(frame)
    # print(newFrame)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()