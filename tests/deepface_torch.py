import cv2
from deepface import DeepFace
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set device mode
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device is available. Using CUDA.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available. Using MPS.")
else:
    device = torch.device("cpu")
    print("No GPU support available. Using CPU.")

face_detector = MTCNN(image_size=160,min_face_size=40,keep_all=True,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)

if device.type != torch.device("mps").type:
  face_detector = MTCNN(image_size=160,min_face_size=40,keep_all=True,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

# Initialize video capture (use webcam)
cap = cv2.VideoCapture(0)

colorBox = (255,255,255)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_copy)
    boxes, probs = face_detector.detect(img)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if boxes is not None:
        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            try:
                if prob > 0.9:
                    x1, y1, x2, y2 = map(int, box)
                    # Crop the frame using the bounding box
                    cropped_frame = frame[y1:y2, x1:x2]

                    cv2.imshow("Cropped Frame", cropped_frame)

                    try:
                        analysis = DeepFace.analyze(cropped_frame, actions=['emotion'], enforce_detection=False)
                        analysis = analysis[0]
                        print(analysis)

                        if analysis["face_confidence"] > 0.85:
                            frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), colorBox, 5)   
                            frame = cv2.putText(frame, f"{analysis['dominant_emotion']} ", (int(box[0]),int(box[3]+35)), cv2.FONT_HERSHEY_SIMPLEX, 1, colorBox,2)
                    except Exception as e:
                        print(f"Emotion detection failed: {e}")
                        
            except:
                print("Failed")

    # Display the resulting frame
    cv2.imshow("Face Detection and Emotion Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()