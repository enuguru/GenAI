"""
Copyright (c) 2025, Sesha Balla
All rights reserved.
"""
 
import os
import io
import time
import glob
from flask import Flask, render_template, request, jsonify,send_file
from openai import OpenAI
 
app = Flask(__name__)
 
audio_directory = os.path.join(os.getcwd(), "audiofiles")
 
# Create the audio directory if it doesn't exist
if not os.path.exists(audio_directory):
    os.makedirs(audio_directory)
 
def delete_old_audio_files(directory):
    files = glob.glob(os.path.join(directory, '*.wav'))  # Adjust this if you save in a different format
    for file in files:
        os.remove(file)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
 
# Define available languages
LANGUAGES = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Spanish": "es"
}
 
# Function to translate text using OpenAI
def translate_text(text, target_language):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
        ]
    )
    return response.choices[0].message.content
 
def save_audio(audio_fp, file_path):
    try:
        with open(file_path, 'wb') as f:
            f.write(audio_fp.read())
    except Exception as e:
        print(f"Error saving audio: {e}")
 
def convert_text_to_speech_openai(article_text, language='en', voice='alloy'):
    try:
        tts = client.audio.speech.with_raw_response.create( # removed with statement as with_raw_response does not support it
            model="tts-1",
            voice=voice,
            input=article_text,
            response_format="wav",  # Specify the desired format
        )
        audio_fp = io.BytesIO(tts.content) # using response.read() to get the audio content
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        print(f"Error converting text to speech using OpenAI: {e}")
        return None
 
@app.route('/', methods=['GET', 'POST'])
def index():
    audio_files = {}
    translations = {}
    char_count = 0
    original_text = ""
    if request.method == 'POST':
        delete_old_audio_files(audio_directory)  # Delete old audio files
        text = request.form['text']
        voice = request.form.get('optionList', 'alloy')  # Get the selected voice from the form
        print(voice)
        original_text = text
        original_audio_fp = convert_text_to_speech_openai(original_text, language='en', voice=voice)
        original_audio_file_path = os.path.join(audio_directory, f'original_{int(time.time())}.wav')
        save_audio(original_audio_fp, original_audio_file_path)
        audio_files['Original'] = original_audio_file_path
        char_count = len(text)
        if char_count > 3000:
            text = text[:3000]
            char_count = 3000
        for lang_name, lang_code in LANGUAGES.items():
            translated_text = translate_text(text, lang_name)
            if translated_text is None:
               error_message = f"Unable to translate to {lang_name}."
            else:
                translations[lang_name] = translated_text
                audio_fp = convert_text_to_speech_openai(translated_text, language=lang_name, voice=voice)
                audio_file_path = os.path.join(audio_directory, f'{lang_name}_{int(time.time())}.wav')
                save_audio(audio_fp, audio_file_path)
                audio_files[lang_name] = audio_file_path    
    return render_template('index.html', languages=LANGUAGES.keys(), translations=translations, char_count=char_count, original_text=original_text, audio_files=audio_files)
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(audio_directory, filename), as_attachment=True)
 
if __name__ == '__main__':
    app.run(debug=True)