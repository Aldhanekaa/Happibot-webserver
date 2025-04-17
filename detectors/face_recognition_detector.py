import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
import tensorflow as tf 
import torch
from PIL import Image
import cv2
import os

try:
    __dir__ = os.path.dirname(os.path.abspath(__file__))
except NameError:
    __dir__ = os.getcwd()

class FaceRecognitionDetector():
    def __init__(self, device):
        self.device = device
        self.image_size = 160
        self.mtcnn = MTCNN(image_size=self.image_size,min_face_size=40,keep_all=True,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
        load_data = torch.load(os.path.join(__dir__, '/detectors/Level11_Embeddings_2.pt'), map_location=device) 
        self.embedding_list = load_data[0] 
        self.name_list = load_data[1] 
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        self.pause = False
        self.frame = None
    
    def detect(self, frame):
        img = Image.fromarray(frame)
        # print(self.name_list)
        img_cropped_list, prob_list = self.mtcnn(img, return_prob=True) 
        results = []
        
        if img_cropped_list is not None:
            img_cropped_list = img_cropped_list.to(self.device)
            boxes, _ = self.mtcnn.detect(img)
            for i, prob in enumerate(prob_list):
                if prob>0.90:
                    emb = self.resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(self.embedding_list):
                        dist = torch.dist(emb.to(device=self.device), emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list) # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                    name = self.name_list[min_dist_idx-1]

                    box = boxes[i] 
                    predicted_name = "Unknown"
                    print(f"Min dist {min_dist}")
                    if (min_dist) < 0.7:
                        min_dist = 1 - min_dist
                        predicted_name = name
                        print(f"{predicted_name} {1- min_dist}")
                    frame = cv2.putText(frame, predicted_name+' '+str((1-min_dist) * 100)+"%", (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)
                        
                        # else:
                        # print(box)
                    results.append([predicted_name, box])                        
                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
            
            return frame