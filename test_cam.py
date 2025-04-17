import cv2
from detectors.face_recognition_detector import FaceRecognitionDetector

cap = cv2.VideoCapture(0)  # Open the camera

FaceRecognition = FaceRecognitionDetector()
while True:
    ret, frame = cap.read()

    newFrame = FaceRecognition.detect(frame)
    print(newFrame)
    
    if not ret or frame is None:
        print("Error: Failed to capture frame!")
        break
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()