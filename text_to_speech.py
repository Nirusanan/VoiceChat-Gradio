from kokoro import KPipeline
from IPython.display import display
import soundfile as sf
import torch
import numpy as np

pipeline = KPipeline(lang_code='a')

def text_to_speech_respond(text):
    final_audio = []
    rate = 24000
    generator = pipeline(text, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        final_audio.append(audio) 
        
    audio = np.concatenate(final_audio, axis=0).astype(np.float32)
    return rate, audio
