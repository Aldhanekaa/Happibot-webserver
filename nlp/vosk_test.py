import vosk
import sounddevice as sd
import json
import queue

# Load Vosk model (Download lightweight models from https://alphacephei.com/vosk/models)
model = vosk.Model("models/vosk-model-small-en-us-0.15")

q = queue.Queue()

def callback(indata, frames, time, status):
    """Audio callback function that adds recorded audio to queue"""
    if status:
        print(status)
    q.put(bytes(indata))

def recognize_speech():
    """Continuously listens for 'Hello Happy Bot' and records speech until silence is detected"""
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        recognizer = vosk.KaldiRecognizer(model, 16000)
        print("Listening for wake word...")
        
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())["text"]
                print(f"Detected: {result}")

                if "hello happy bot" in result:
                    print("Wake word detected! Listening for further input...")
                    
                    while True:
                        data = q.get()
                        if recognizer.AcceptWaveform(data):
                            speech_result = json.loads(recognizer.Result())["text"]
                            if speech_result.strip():
                                print(f"You said: {speech_result}")
                            else:
                                print("Silence detected. Stopping...")
                                return

recognize_speech()
