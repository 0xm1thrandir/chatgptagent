import openai, config, subprocess
import gradio as gr
from flask import Flask, request

openai.api_key = config.OPENAI_API_KEY
messages = [{"role": "system", "content": 'You are a therapist. Respond to all input in 25 words or less.'}]

app = Flask(__name__)

@app.route("/", methods=["POST"])
def chat():
    global messages

    # Receive audio file
    file = request.files["audio"]

    # Transcribe audio using OpenAI API
    audio_file = file.read()
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Add user message to messages list
    messages.append({"role": "user", "content": transcript["text"]})

    # Generate AI response using OpenAI API
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    system_message = response["choices"][0]["message"]

    # Add AI response to messages list
    messages.append(system_message)

    # Speak AI response
    subprocess.call(["say", system_message['content']])

    # Generate chat transcript
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

# Create a Gradio interface to test the app locally
interface = gr.Interface(fn=chat, inputs=gr.Audio(source="microphone", type="file"), outputs="text")
interface.test_launch()

# Uncomment the following line to launch the app in a browser window
# interface.launch()

