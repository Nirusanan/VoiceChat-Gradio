import os
import json
from groq import Groq


client = Groq(api_key = os.getenv('GROQ_API_KEY'))
model = 'whisper-large-v3'


def transcribe_and_respond(audio_path):    
    with open(audio_path, "rb") as file:
    # Create a transcription of the audio file
        transcription = client.audio.transcriptions.create(
        file=file, 
        model="whisper-large-v3", 
        prompt="Specify context or spelling",  
        response_format="verbose_json",  
        timestamp_granularities = ["word", "segment"], 
        language="en",  
        temperature=0.0  
        )

    return transcription.text