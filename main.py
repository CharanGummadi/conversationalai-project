from flask import Flask, render_template, request, jsonify, send_file
from google.cloud import speech, texttospeech, language_v1
import os
import io
import uuid

app = Flask(__name__)

# Load Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/charan/app/conversationalai-436500-5f0cbecce377.json"

# Create a folder to store transcriptions
TRANSCRIPTIONS_FOLDER = 'transcriptions'
if not os.path.exists(TRANSCRIPTIONS_FOLDER):
    os.makedirs(TRANSCRIPTIONS_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_content = audio_file.read()
        app.logger.info(f"Received audio data of size: {len(audio_content)} bytes")

        # Google Speech-to-Text API call
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)
        
        if not response.results:
            return jsonify({'error': 'No transcription results'}), 404

        transcription = ' '.join([result.alternatives[0].transcript for result in response.results])
        app.logger.info(f"Transcription: {transcription}")

        # Perform sentiment analysis
        sentiment = analyze_sentiment(transcription)

        # Save transcription and sentiment to file
        file_path = save_transcription(transcription, sentiment)

        return jsonify({
            'transcription': transcription,
            'sentiment': sentiment,
            'file_path': file_path
        })

    except Exception as e:
        app.logger.error(f"Error in upload_audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert_text', methods=['POST'])
def convert_text():
    try:
        data = request.get_json()
        text = data['text']

        # Perform sentiment analysis
        sentiment = analyze_sentiment(text)

        # Save text and sentiment to file
        file_path = save_transcription(text, sentiment, is_text=True)

        # Google Text-to-Speech API call
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

        return response.audio_content, 200, {
            'Content-Type': 'audio/mp3',
            'X-Sentiment': sentiment,
            'X-File-Path': file_path
        }

    except Exception as e:
        app.logger.error(f"Error in convert_text: {str(e)}")
        return jsonify({'error': str(e)}), 500

def analyze_sentiment(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment

    if sentiment.score > 0.25:
        return 'Positive'
    elif sentiment.score < -0.25:
        return 'Negative'
    else:
        return 'Neutral'

def save_transcription(content, sentiment, is_text=False):
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}.txt"
    file_path = os.path.join(TRANSCRIPTIONS_FOLDER, file_name)

    with open(file_path, 'w') as f:
        f.write(f"{'Text' if is_text else 'Transcription'}:\n{content}\n\nSentiment: {sentiment}")

    return f"/transcriptions/{file_name}"

@app.route('/transcriptions/<path:filename>')
def serve_transcription(filename):
    return send_file(os.path.join(TRANSCRIPTIONS_FOLDER, filename))

if __name__ == "__main__":
    app.run(debug=True)