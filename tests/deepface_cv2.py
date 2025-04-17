import cv2
from deepface import DeepFace
import torch

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize video capture (use webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face region for emotion detection
        face_crop = frame[y:y + h, x:x + w]
        face_crop_tensor = torch.tensor(face_crop)
        print(face_crop_tensor.shape)

        try:
            # Analyze the cropped face for emotion
            analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            analysis = analysis[0]
            print(analysis)
            # Extract the dominant emotion
            dominant_emotion = analysis['dominant_emotion']

            # Display the detected emotion on the frame
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Emotion detection failed: {e}")

    # Display the resulting frame
    cv2.imshow("Face Detection and Emotion Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()