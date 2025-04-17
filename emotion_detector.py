import threading
import cv2
import tensorflow as tf 
from torch.nn.functional import interpolate
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN
import numpy as np
import torch
from PIL import Image


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

class EmotionDetector():
    def __init__(self, cap, happiness_states_queue,device):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = False  # Make the thread a daemon thread
        self.cap = cap
        self.happiness_states_queue = happiness_states_queue
        self.image_size = 160
        self.mtcnn = MTCNN(image_size=self.image_size,min_face_size=40,margin=20,post_process=False,keep_all=True )
        with tf.device("/GPU:0"):
            self.happiness_detection_model = tf.keras.models.load_model("./Happiness Detection Model.keras")
        self.pause = True
    
    def run(self) :
        while self.cap.isOpened() :
            ret, frame = self.cap.read()
            # print(f"Pause Processing Happiness Detection : {self.pause}")
            if self.pause:
                continue
            if not ret:
                print("fail to grab frame, try again")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            img_cropped_list, prob_list = self.mtcnn(frame, return_prob=True) 
            results = []
            results.append([])
            states = []
            if img_cropped_list is not None:
                boxes, _ = self.mtcnn.detect(frame)
                        
                for i, prob in enumerate(prob_list):
                    if prob>0.90:
                        pre_box = boxes[i]
                        self.box = pre_box

                        margin = 20
                        margin = [
                            margin * (pre_box[2] - pre_box[0]) / (self.image_size - margin),
                            margin * (pre_box[3] - pre_box[1]) / (self.image_size - margin),
                        ]
                        raw_image_size = get_size(img)
                        box = [
                            int(max(pre_box[0] - margin[0] / 2, 0)),
                            int(max(pre_box[1] - margin[1] / 2, 0)),
                            int(min(pre_box[2] + margin[0] / 2, raw_image_size[0])),
                            int(min(pre_box[3] + margin[1] / 2, raw_image_size[1])),
                        ]

                        face = crop_resize(img, box, self.image_size)
                        face = F.to_tensor(face)
                        face = transforms.Grayscale().forward(face)
                        p = interpolate(face.unsqueeze(0), size=(48, 48), mode='bilinear', align_corners=False).squeeze(0)
                        p = p.permute(1,2,0)
                        p = p.unsqueeze(0)
                        p = p.numpy()
                        happiness_outputs = self.happiness_detection_model.predict(p)
                        happiness_outputs = happiness_outputs[0]
                        self.happiness_state = ""
                        self.happiness_output = 0
                        self.happiness_colour = (0,0,0)
                        if happiness_outputs[1] > 0.5:
                            self.happiness_state = "Happy"
                            self.happiness_output = happiness_outputs[1]
                            self.happiness_colour = (0,255,0)
                        elif happiness_outputs[0] > happiness_outputs[1]:
                            self.happiness_state = "Not Happy"
                            self.happiness_output = happiness_outputs[0]
                            self.happiness_colour = (255,0,0)
                        else:
                            self.happiness_state = "Happy"
                            self.happiness_output = happiness_outputs[1]
                            self.happiness_colour = (0,255,0)
                        states.append([self.box, self.happiness_state, self.happiness_output, self.happiness_colour])
                
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # results[0] = frame
            results.append(states)
            self.happiness_states_queue.put(results)
    
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