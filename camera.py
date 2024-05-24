import threading
import cv2
class Camera():
    def __init__(self, cap):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = False  # Make the thread a daemon thread
        self.cap = cap
        self.frame = None
        self.ret = None

    def getFrame(self):
        return self.frame
    def run(self) :
        if not self.cap.isOpened():
            self.cap.open(0)
    
        while self.cap.isOpened():
            # print("HEYY")
            self.ret, self.frame = self.cap.read()
    
    def start(self):
        self.thread.start()
    def isOpened(self):
        return self.cap.isOpened()

    def join(self):
        self.thread.join()
    def stop(self):
        self.thread._stop()
            
    