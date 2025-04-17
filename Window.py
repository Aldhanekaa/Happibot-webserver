import threading
import cv2

import face_recognition

class Window():
    def __init__(self, faceDetector: face_recognition.FaceRecognition):
        self.thread = threading.Thread(target=self.run)
        self.faceDetector = faceDetector

    def run(self):
        while self.faceDetector.pause is False:
            if self.faceDetector.frame is None:
                continue

            print(self.faceDetector.frame)
            cv2.imshow("Face", self.faceDetector.frame)
    def start(self):
        self.thread.start()

    def stop(self):
        self.thread._stop()
        self.thread.join()