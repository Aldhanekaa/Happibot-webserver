from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_wav("failed_generate_audio.wav")
play(audio)