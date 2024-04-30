import os
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
from tensorflow.keras.models import model_from_json

# Initialize Flask application
app = Flask(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Define Singleton for the model
class SingletonModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with open('CNN_model.json', 'r') as json_file:
                loaded_model_json = json_file.read()
                cls._instance = model_from_json(loaded_model_json)
                cls._instance.load_weights("best_model1_weights.h5")

            with open('scaler2.pickle', 'rb') as f:
                cls.scaler = pickle.load(f)

        return cls._instance

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract features from audio data
def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    rmse1 = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    features = np.hstack((np.squeeze(zcr), np.squeeze(rmse1), np.ravel(mfcc.T)))
    return features

# Predict emotion label
def predict_emotion(data):
    model = SingletonModel()
    features = extract_features(data)
    scaled_features = model.scaler.transform(features.reshape(1, -1))
    result = model.predict(np.expand_dims(scaled_features, axis=2))
    emotion_label = np.argmax(result)
    return emotion_label

# Route to predict emotion
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No uploaded file'})

    # Check if the file format is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # Process the audio file and predict emotion
    try:
        data, _ = librosa.load(file.stream, duration=2.5, offset=0.6)
        emotion_label = predict_emotion(data)
        emotion = emotions1[emotion_label]
        return jsonify({'success': True, 'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)})
