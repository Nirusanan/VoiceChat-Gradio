from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np

pipeline = KPipeline(lang_code='a')

text = '''
Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

This is highly serias activity. So, I will try to complete this one.
'''

final_audio = [] 

# voice = ['af_heart', 'af_bella', 'af_sarah', 'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george']
generator = pipeline(text, voice='bf_isabella')

for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    final_audio.append(audio) 
    
audio = np.concatenate(final_audio, axis=0).astype(np.float32)
display(Audio(data=audio, rate=24000, autoplay=True)) # Display the 24khz audio
sf.write(f'final.wav', audio, 24000)