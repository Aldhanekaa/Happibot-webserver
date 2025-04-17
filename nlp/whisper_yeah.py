import whisper
import numpy as np
import sounddevice as sd
import re

# Load Whisper model (Tiny for faster response, Medium/Large for better accuracy)
model = whisper.load_model("small.en") # "small.en"

pattern = r"\b(happy\s?bot|happy\s?bird|happy\s?work|happy\s?birthday)\b"

def contains_happy_variations(text):
    """Checks if the text contains 'happybot', 'happy bot', 'happybird', 'happy bird', or 'happy work'"""
    return bool(re.search(pattern, text, re.IGNORECASE))


def record_audio(duration=10, samplerate=16000):
    """Records audio for a given duration."""
    print("Listening...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    return np.squeeze(audio)

def transcribe_audio(audio):
    """Transcribes recorded audio using Whisper."""
    result = model.transcribe(audio, fp16=False)
    return result["text"]

def listen_for_wake_word():
    """Continuously listens for wake word before activating full speech recognition."""
    while True:
        audio = record_audio(duration=4)  # Short listening window
        text = transcribe_audio(audio).lower()
        print(f"Detected:{text}")
        result = contains_happy_variations(text)
        print(f"result:{result}")

        if result:
            print("Wake word detected! Listening for further input...")
            full_audio = record_audio(duration=10)  # Listen for a longer duration
            final_text = transcribe_audio(full_audio)
            print(f"You said: {final_text}")
            break  # Stop after full speech input

listen_for_wake_word()