import threading
from facenet_pytorch import MTCNN, InceptionResnetV1
import tensorflow as tf 
import torch
from PIL import Image
import cv2

class FaceRecognition():
    def __init__(self, cap, faces_queue, device):
        self.device = device
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = False  # Make the thread a daemon thread
        self.cap = cap
        self.faces_queue = faces_queue
        self.image_size = 160
        self.mtcnn = MTCNN(image_size=self.image_size,min_face_size=40,keep_all=True,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
        load_data = torch.load('EmbeddingsTrain_Level10.pt',map_location=device) 
        self.embedding_list = load_data[0] 
        self.name_list = load_data[1] 
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        self.pause = False
        self.frame = None
    
    def run(self) :
        device = self.device
        while self.cap.isOpened() :
            ret, frame = self.cap.read()

            print(f"Pause Processing Face Recognition : {self.pause}")
            if self.pause:
                continue
            if not ret:
                print("fail to grab frame, try again")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_cropped_list, prob_list = self.mtcnn(img, return_prob=True) 
    
        
            results = []
            if img_cropped_list is not None:
                img_cropped_list = img_cropped_list.to(device)
                boxes, _ = self.mtcnn.detect(img)
                        
                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        emb = self.resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                        
                        dist_list = [] # list of matched distances, minimum distance is used to identify the person
                        
                        for idx, emb_db in enumerate(self.embedding_list):
                            dist = torch.dist(emb.to(device=device), emb_db).item()
                            dist_list.append(dist)

                        min_dist = min(dist_list) # get minumum dist value
                        min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                        name = self.name_list[min_dist_idx-1] # get name corrosponding to minimum dist
                        
                        box = boxes[i] 
                        predicted_name = "Unknown"
                        

                        if min_dist<0.65:
                            min_dist = 1 - min_dist
                            predicted_name = name
                            print(f"{name} {1- min_dist}")
                        frame = cv2.putText(frame, name+' '+str((1-min_dist) * 100)+"%", (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)
                        
                        # else:
                        # print(box)
                        results.append([predicted_name, box])                        
                        frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                        self.frame = frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
            # cv2.imshow("face recog",frame)
            # print(results)
            self.faces_queue.put(results)
    def start(self):
        self.thread.start()
    
    def pauseRunning(self):
        self.pause = True
    def resumeRunning(self):
        if self.pause == True:
            self.pause = False
            
    def join(self):
        self.thread.join()
    def stop(self):
        self.thread._stop()
        self.thread.join()