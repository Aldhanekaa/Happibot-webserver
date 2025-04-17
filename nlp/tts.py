from TTS.api import TTS
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device is available:", device)
else:
    print("MPS device is not available. Using CPU instead.")
    device = torch.device("cpu")


# Load a pre-trained TTS model
tts = TTS(model_name="tts_models/en/ljspeech/speedy-speech", progress_bar=True).to(device=device)

# Generate speech and save to file
tts.tts_to_file(text="That's a fun challenge!  What kind of things does your Valentine like?  Maybe we can brainstorm some ideas together.",  file_path="speedy.wav")
