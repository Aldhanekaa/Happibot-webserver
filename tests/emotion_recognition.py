import os
import cv2
import torch
import tensorflow as tf
import numpy as np
# from keras.api.models import load_model
from keras._tf_keras.keras.models import load_model
from emotion_recognition_net import build_emotionRecognition_net
from facenet_pytorch import MTCNN
from PIL import Image
import torch.nn.functional as F

# from tensorflow.keras.models import model_from_yaml

import torch


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


devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

current_dir = os.getcwd()
model_weights_path = os.path.join(current_dir, "models/AFHSN_EmotionRecognition_Model.h5")
model_yaml_path = os.path.join(current_dir, "models/AFHSN_EmotionRecognition_Model.yaml")
model_json_path = os.path.join(current_dir, "models/AFHSN_EmotionRecognition_Model.json")

# MODEL CLASSES CONFIGURATION
emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
INTERESTED_LABELS = [0, 2, 3, 4, 6]
emotions = {}
num_classes = len(INTERESTED_LABELS)

print("Current working directory:", current_dir)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)

model = build_emotionRecognition_net(num_classes=num_classes)

model.load_weights(model_weights_path)
print(model)

# MODEL CLASSES CONFIGURATION
for i in range(len(INTERESTED_LABELS)):
    emotions[i] = emotion_label_to_text[INTERESTED_LABELS[i]]

print(emotions)
if model is None:
  print("Fail to load model")

# Open the default camera (usually 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read and display frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_copy)
    boxes, probs = face_detector.detect(img)


    if boxes is not None:
      for idx, (box, prob) in enumerate(zip(boxes, probs)):
        try:
          if prob > 0.9:
            colorBox = (255,255,255)

            # print(box)
            # print(idx)
            x1, y1, x2, y2 = map(int, box)
            # Crop the frame using the bounding box
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

            img_tensor = torch.tensor(cropped_frame, device=device)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor: torch.Tensor = F.interpolate(img_tensor, size=(48, 48), mode='bilinear', align_corners=False)
            img_tensor = img_tensor.squeeze(1)
            img_tensor = img_tensor.permute(1,2,0)
            # print(img_tensor.shape) # 48,48,1
            

            cropped_img = img_tensor.cpu().numpy()
            # Normalize the image values to the range [0, 255] (if needed)
            cropped_img = np.uint8((cropped_img - cropped_img.min()) / (cropped_img.max() - cropped_img.min()) * 255)

            # Show the image with OpenCV
            cv2.imshow("48x48 Image", cropped_img)

            try:
              img_tensor = img_tensor.permute(2,0,1)
              logits = model.predict(img_tensor.cpu().numpy(), batch_size=32)
              classes_x=np.argmax(logits,axis=1)
              

              print("LOGITS")
              print(logits)

              emotion = emotions[classes_x[0]]
              accuracy = logits[0][classes_x[0]]

              if logits[0][3] > 0.8:
                emotion = "sadness"
                accuracy = logits[0][3]
              
              if (logits[0][0] > 0.7 and logits[0][2] > 0.7) or (logits[0][0] > 0.7):
                emotion = "anger"
                accuracy = logits[0][0]

              print("CLASSES ")
              # print(classes_x)
              if emotion == "anger":
                colorBox = (0,0,255)
              elif emotion == "happiness":
                colorBox = (0,255,0)

              print("PERCENTAGE : ")
              for emotion_idx in emotions:
                print(f"Accuracy of {emotions[emotion_idx]} : {logits[0][emotion_idx]* 100:.1f}%")
                
              frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), colorBox, 5)   
              frame = cv2.putText(frame, f"{emotion} {accuracy * 100:.1f}%", (int(box[0]),int(box[3]+35)), cv2.FONT_HERSHEY_SIMPLEX, 1, colorBox,2)
            except Exception as e:
              print(f"Error {e}")

            # Display the cropped frame
            cv2.imshow("Cropped Frame", cropped_frame)
        except:
          print("Failed asset")

    # Display the frame in a window
    cv2.imshow('Camera', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
