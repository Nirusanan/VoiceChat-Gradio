import os
import json
from groq import Groq
import gradio as gr
from speech_to_text import transcribe_and_respond 


client = Groq(api_key = os.getenv('GROQ_API_KEY'))



def voice_respond(audio_path):
    prompt = transcribe_and_respond(audio_path)

    template = "Respond concisely and professionally. Do not use phrases like 'I can assist you' or similar. Instead, directly provide the required information.\n" + prompt

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": template,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    llm_response = chat_completion.choices[0].message.content

    return llm_response

    
    

demo = gr.Interface(
    fn=voice_respond,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=[gr.Textbox(label="LLM Response")],
    title="Voice Chat with Whisper & Llama",
    description="Speak into your microphone, transcribe speech to text using Whisper, and get a response from an LLM."
)

demo.launch()