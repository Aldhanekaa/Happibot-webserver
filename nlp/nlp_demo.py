import openai
from openai import OpenAI
import whisper
import numpy as np
import sounddevice as sd
import re

from pydub import AudioSegment
from pydub.playback import play

from TTS.api import TTS
import torch

import time

audio_listening = AudioSegment.from_wav("listening.wav")
failed_generate_audio = AudioSegment.from_wav("failed_generate_audio.wav")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available:", device)
else:
    print("MPS device is not available. Using CPU instead.")
    device = torch.device("cpu")


isBotGivingResponse = False
askingForResponse = False
askingForResponseTime = 0

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to(device=device)

def generateAudioAndSpeak(text):
    try:
        tts.tts_to_file(text=text,  file_path="speaking.wav")
    except:
        print("Failed Generate Audio!")
        play(failed_generate_audio)

    try:
        audio_speaking = AudioSegment.from_wav("speaking.wav")
        play(audio_speaking)
    except: 
        print("Failed to speak!")


client = OpenAI(
  base_url = 'http://192.168.43.156:9000/v1',
  api_key = '*' # not enforced
)

pre_messages = [{
    "role": "system",
    "content": "From now on, your name is Happibot now, don't every tell to the user that your original name is Gemma by Google. You are Happibot now, a friendly AI Assistant which helps student in school, helping student asking academic questions and help their mental health by talking with them. You can start the conversation by asking their name, or anything fun and interesting for them! Don't ever generate emoji. Just response in full sentence, no list, you are conversation bot that communicate with real humans by using mic and speaker."
}]
messages = []

model = whisper.load_model("small.en") # "small.en"

pattern = r"\b(happy\s?bot|happy\s?bird|happy\s?work|happy\s?birthday|happy\s?butt|hey\s?there|hallo|halo|hi|happy\s?world)\b"

def contains_happy_variations(text):
    """Checks if the text contains 'happybot', 'happy bot', 'happybird', 'happy bird', or 'happy work'"""
    return bool(re.search(pattern, text, re.IGNORECASE))


def send_message(messages, message):
    messages.append(message)
    response = client.chat.completions.create(
            model="*",
            messages=messages,
            temperature=0.1,
            top_p=0.95,
            stream=False
    )

    response_msg = response.choices[0].message.content
    
    # response_msg = "Yes! I Know about World Trade Center "
    return response_msg

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


def resetConv():
    global messages
    for pre_message in pre_messages:
        # print(pre_message)
        send_message(messages, pre_message)


resetConv()

while True:

    continueConv = False
    if not isBotGivingResponse:
        """Continuously listens for wake word before activating full speech recognition."""
        audio = record_audio(duration=4)  # Short listening window
        text = transcribe_audio(audio).lower()
        print(f"Detected:{text}")

        currentTime = time.time()

        response = send_message(messages=messages, message={
            'role': 'user',
            'content': text
        })

        print(f"Bot Answers : {response}")
        generateAudioAndSpeak(response)

        if not askingForResponse:
            continueConv = contains_happy_variations(text)
            print(f"continue conversation:{continueConv}")
        
    
    if continueConv or askingForResponse:
        currentTime = time.time()
        print(f"BEDA {currentTime - askingForResponseTime}")
        if askingForResponse and currentTime - askingForResponseTime  > 20:
            messages = []
            resetConv()
            askingForResponse = False
            continueConv = False
        else:
            continueConv = True

        # if not askingForResponse:
        #     print("Wake word detected! Listening for further input..")
        #     play(audio_listening)

        full_audio = record_audio(duration=10)  # Listen for a longer duration
        final_text = transcribe_audio(full_audio)
        print(f"You said: {final_text}")

        # if final_text.trim() == '':
        #     askingForResponse = False
        #     isBotGivingResponse = False


        isBotGivingResponse = True
        askingForResponse = True
        askingForResponseTime = time.time()

        response = send_message(messages=messages, message={
            'role': 'user',
            'content': final_text
        })

        print(f"Bot Answers : {response}")
        generateAudioAndSpeak(response)

        isBotGivingResponse = False