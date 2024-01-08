import librosa
import librosa.display 
import os

audio_file = "FASSounds - Morning Flight.mp3"
y, sr = librosa.load(audio_file)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)
