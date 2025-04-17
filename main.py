from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask import request
from camera import Camera
from emotion_detector import EmotionDetector
from face_recognition import FaceRecognition
import atexit
from facenet_pytorch import MTCNN, InceptionResnetV1

import cv2
import base64
from queue import Queue
import threading
import torch


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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

image_size = 160

# resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

happiness_states_queue = Queue(maxsize=0)
faces_queue = Queue(maxsize=0)

cam =  Camera(cap)
# emotionDetector =  EmotionDetector(cap,happiness_states_queue, device=device)
faceDetector =  FaceRecognition(cap,faces_queue, device=device)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for React app
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

active_sessions = []
user_threads = {}

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def capture_frames(userId, stop_event):
    print(f"CAPTURE {userId}")
    face_results_saved = []
    try:
        while not stop_event.is_set():
            
            # print(f"TYPE {type(frame)}")
            # frame = cam.getFrame()
            # _, buffer = cv2.imencode('.jpg', frame)
            # frame_data = base64.b64encode(buffer).decode('utf-8')
            # socketio.emit('frame', frame_data)
            # socketio.sleep(0.1)  # sleep to limit the frame rate
            
            try:
                frame = cam.getFrame()

                if frame is None:
                    continue
                
                if not happiness_states_queue.empty():
                    results = happiness_states_queue.get()
                    face_results = None

                    # print(f"tye {type(frame)}")

                    for i in range(len(results[1])):
                        result = results[1][i]
                        box = result[0]
                        happiness_state = result[1]
                        happiness_output = result[2]
                        happiness_colour = result[3]

                        frame = cv2.putText(frame, f"{happiness_state} {happiness_output*100:.2f}%", (int(box[0]),int(box[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, happiness_colour,2)
                        frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), happiness_colour, 5)            
                
                if not faces_queue.empty():
                    face_results = None
                    face_results = faces_queue.get()
                    face_results_saved = face_results
                    # print("Face Results")
                    # print(face_results)
                    print("Face Results Saved")
                    print(face_results)
                    print(len(face_results_saved))


                    if face_results_saved is not None:
                        i = 0
                        for i in range(len(face_results_saved)):
                            print(i)
                            print(f"face {i}")
                            box = face_results_saved[i][1]
                            print("box")
                            print(box)
                            frame = cv2.putText(frame, face_results_saved[i][0], (int(box[0]),int(box[3]+35)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('frame', frame_data)
                    socketio.sleep(0.1)  # sleep to limit the frame rate
                    
                socketio.sleep(0.1) 
                
                # _, buffer = cv2.imencode('.jpg', frame)
                # frame_data = base64.b64encode(buffer).decode('utf-8')
                # socketio.emit('frame', frame_data)
                # socketio.sleep(0.1)  # sleep to limit the frame rate

            except:
                pass
                # print("YE")
    except:
        pass
        # print("SH")

@app.route('/')
def index():
    return "WebSocket Server Running"

@socketio.on('connect')
def handle_connect():
    active_sessions.append(request.sid)
    print(f'Client connected {request.sid}')

    user_id = request.sid

    if user_id not in user_threads:
        stop_event = threading.Event()
        thread = threading.Thread(target=capture_frames, args=(request.sid, stop_event))
        thread.daemon = True
        thread.start()
        user_threads[user_id] = (thread, stop_event)
        # emotionDetector.resumeRunning()
        faceDetector.resumeRunning()
    # socketio.start_background_task(capture_frames)

@socketio.on('disconnect')
def handle_disconnect():
    active_sessions.remove(request.sid)
    print(f'Client disconnected {request.sid}')

    user_id = request.sid

    if user_id in user_threads:
        # print(f"sTOP! {user_id}")
        thread, stop_event = user_threads[user_id]
        stop_event.set()
        thread.join()  # Wait for the thread to finish
        # print(f"FINISHED {user_id}")
        del user_threads[user_id]

        if len(user_threads.keys()) == 0:
            happiness_states_queue.empty()
            faces_queue.empty()
            # emotionDetector.pauseRunning()
            faceDetector.pauseRunning()


def exit_handler():
    print("Program exiting...")
    cam.stop()

if __name__ == '__main__':
    cam.start()
    faceDetector.start()
    # emotionDetector.start()
    socketio.run(app, host='0.0.0.0', port=3001)
    atexit.register(exit_handler)
